use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::BergamotEngine;
use crate::catalog::{
    CatalogSnapshot, DeletePlan, DownloadPlan, FsPackInstallChecker, LanguageAvailabilityRow,
    LanguageOverview, PackInstallChecker, build_catalog_snapshot, build_language_overview,
    language_rows_in_snapshot, parse_and_validate_catalog, plan_delete_dictionary,
    plan_delete_language, plan_delete_superseded_tts, plan_delete_support_by_kind, plan_delete_tts,
    plan_dictionary_download, plan_language_download, plan_support_download_by_kind,
    plan_tts_download, select_best_catalog,
};
use crate::routing::MixedTextTranslationResult;
use crate::settings::BackgroundMode;
use crate::styled::{OverlayScreenshot, StructuredTranslationResult, StyledFragment};
#[cfg(feature = "odt")]
use crate::translate::TranslationWithAlignment;
use crate::translate::Translator;

#[cfg(feature = "dictionary")]
use crate::tarkka::{
    DictionaryCache, WordWithTaggedEntries, close_dictionary_in_snapshot,
    lookup_dictionary_in_snapshot,
};

#[cfg(feature = "tesseract")]
use crate::ocr::{PreparedImageOverlay, ReadingOrder};
#[cfg(feature = "tesseract")]
use crate::ocr_runtime::{OcrCache, translate_image_rgba_in_snapshot};

#[cfg(feature = "tts")]
use crate::api::VoiceName;
#[cfg(feature = "tts")]
use crate::speech::{
    SpeechCache, available_tts_voices_in_snapshot, plan_speech_chunks_for_text_in_snapshot,
    synthesize_pcm_in_snapshot, warm_tts_model_in_snapshot,
};
#[cfg(feature = "tts")]
use crate::tts::{PcmAudio, SpeechChunk, TtsVoiceOption};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum Feature {
    Core,
    Dictionary,
    Tts,
}

static BERGAMOT_ENGINE: OnceLock<Mutex<BergamotEngine>> = OnceLock::new();

pub struct TranslatorSession {
    snapshot: RwLock<Arc<CatalogSnapshot>>,
    #[cfg(feature = "tts")]
    speech: Mutex<SpeechCache>,
    #[cfg(feature = "dictionary")]
    dictionaries: Mutex<DictionaryCache>,
    #[cfg(feature = "tesseract")]
    ocr: Mutex<OcrCache>,
}

impl TranslatorSession {
    pub fn from_snapshot(snapshot: CatalogSnapshot) -> Self {
        Self {
            snapshot: RwLock::new(Arc::new(snapshot)),
            #[cfg(feature = "tts")]
            speech: Mutex::new(SpeechCache::new()),
            #[cfg(feature = "dictionary")]
            dictionaries: Mutex::new(DictionaryCache::new()),
            #[cfg(feature = "tesseract")]
            ocr: Mutex::new(OcrCache::new()),
        }
    }

    pub fn from_catalog(catalog: crate::catalog::LanguageCatalog, base_dir: String) -> Self {
        let checker = FsPackInstallChecker::new(&base_dir);
        let snapshot = build_catalog_snapshot(catalog, base_dir, &checker);
        Self::from_snapshot(snapshot)
    }

    fn engine(&self) -> &'static Mutex<BergamotEngine> {
        BERGAMOT_ENGINE.get_or_init(|| Mutex::new(BergamotEngine::new()))
    }

    pub fn open<C>(
        bundled_json: &str,
        disk_json: Option<&str>,
        base_dir: String,
        install_checker: &C,
    ) -> Result<Self, TranslatorError>
    where
        C: PackInstallChecker,
    {
        let catalog = parse_selected_catalog(bundled_json, disk_json).ok_or_else(|| {
            TranslatorError::new(
                crate::api::TranslatorErrorKind::InvalidInput,
                "failed to parse any catalog",
            )
        })?;
        let snapshot = build_catalog_snapshot(catalog, base_dir, install_checker);
        Ok(Self::from_snapshot(snapshot))
    }

    pub fn snapshot(&self) -> Arc<CatalogSnapshot> {
        self.snapshot
            .read()
            .expect("snapshot lock poisoned")
            .clone()
    }

    fn set_snapshot(&self, snapshot: CatalogSnapshot) {
        *self.snapshot.write().expect("snapshot lock poisoned") = Arc::new(snapshot);
    }

    pub fn refresh_snapshot(&self) {
        let current = self.snapshot();
        let catalog = current.catalog.clone();
        let base_dir = current.base_dir.clone();
        drop(current);
        let checker = FsPackInstallChecker::new(&base_dir);
        self.set_snapshot(build_catalog_snapshot(catalog, base_dir, &checker));
    }

    pub fn apply_delete_plan(&self, plan: &DeletePlan) {
        use std::collections::HashSet;
        use std::path::Path;

        let base_dir = self.snapshot().base_dir.clone();
        let base = Path::new(&base_dir);

        let mut files = plan
            .file_paths
            .iter()
            .map(|path| base.join(path))
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        for file in files {
            let _ = std::fs::remove_file(file);
        }

        let mut directories = plan
            .directory_paths
            .iter()
            .map(|path| base.join(path))
            .collect::<Vec<_>>();
        directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
        let mut seen = HashSet::new();
        for directory in directories {
            if seen.insert(directory.clone()) {
                let _ = std::fs::remove_dir_all(directory);
            }
        }

        self.refresh_snapshot();
    }

    pub fn language_rows(&self) -> Vec<LanguageAvailabilityRow> {
        language_rows_in_snapshot(&self.snapshot())
    }

    pub fn language_overview(&self) -> Vec<LanguageOverview> {
        build_language_overview(&self.snapshot())
    }

    pub fn warm(&self, from_code: &str, to_code: &str) -> Result<(), TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap)
            .warm(&LanguageCode::from(from_code), &LanguageCode::from(to_code))
    }

    pub fn translate_text(
        &self,
        from_code: &str,
        to_code: &str,
        text: &str,
    ) -> Result<String, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_text(
            &LanguageCode::from(from_code),
            &LanguageCode::from(to_code),
            text,
        )
    }

    pub fn translate_html_fragments(
        &self,
        from_code: &str,
        to_code: &str,
        fragments: &[String],
    ) -> Result<Vec<String>, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_html_fragments(
            &LanguageCode::from(from_code),
            &LanguageCode::from(to_code),
            fragments,
        )
    }

    pub fn translate_mixed_texts(
        &self,
        inputs: &[String],
        forced_source_code: Option<&str>,
        target_code: &str,
        available_language_codes: &[LanguageCode],
    ) -> Result<MixedTextTranslationResult, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_mixed_texts(
            inputs,
            forced_source_code.map(LanguageCode::from).as_ref(),
            &LanguageCode::from(target_code),
            available_language_codes,
        )
    }

    pub fn translate_structured_fragments(
        &self,
        fragments: &[StyledFragment],
        forced_source_code: Option<&str>,
        target_code: &str,
        available_language_codes: &[LanguageCode],
        screenshot: Option<&OverlayScreenshot>,
        background_mode: BackgroundMode,
    ) -> Result<StructuredTranslationResult, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_structured_fragments(
            fragments,
            forced_source_code.map(LanguageCode::from).as_ref(),
            &LanguageCode::from(target_code),
            available_language_codes,
            screenshot,
            background_mode,
        )
    }

    pub fn translate_structured_fragments_batch(
        &self,
        pages: &[&[StyledFragment]],
        forced_source_code: Option<&str>,
        target_code: &str,
        available_language_codes: &[LanguageCode],
        background_mode: BackgroundMode,
    ) -> Result<Vec<StructuredTranslationResult>, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_structured_fragments_batch(
            pages,
            forced_source_code.map(LanguageCode::from).as_ref(),
            &LanguageCode::from(target_code),
            available_language_codes,
            background_mode,
        )
    }

    #[cfg(feature = "odt")]
    pub(crate) fn translate_texts_with_alignment(
        &self,
        from_code: &LanguageCode,
        to_code: &LanguageCode,
        texts: &[String],
    ) -> Result<Option<Vec<TranslationWithAlignment>>, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap)
            .translate_texts_with_alignment(from_code, to_code, texts)
    }

    #[cfg(feature = "tesseract")]
    pub fn translate_image_rgba(
        &self,
        rgba_bytes: &[u8],
        width: u32,
        height: u32,
        source_code: &str,
        target_code: &str,
        min_confidence: u32,
        reading_order: ReadingOrder,
        background_mode: BackgroundMode,
    ) -> Result<PreparedImageOverlay, TranslatorError> {
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        let mut ocr = self.ocr.lock().expect("ocr cache poisoned");
        translate_image_rgba_in_snapshot(
            &mut engine,
            &mut ocr,
            &snap,
            rgba_bytes,
            width,
            height,
            &LanguageCode::from(source_code),
            &LanguageCode::from(target_code),
            min_confidence,
            reading_order,
            background_mode,
        )
    }

    pub fn plan_download(
        &self,
        language_code: &str,
        feature: Feature,
        selected_tts_pack_id: Option<&str>,
    ) -> Option<DownloadPlan> {
        let snap = self.snapshot();
        let code = LanguageCode::from(language_code);
        match feature {
            Feature::Core => Some(plan_language_download(&snap, &code)),
            Feature::Dictionary => plan_dictionary_download(&snap, &code),
            Feature::Tts => plan_tts_download(&snap, &code, selected_tts_pack_id),
        }
    }

    pub fn plan_support_download_by_kind(&self, support_kind: &str) -> Option<DownloadPlan> {
        let snap = self.snapshot();
        plan_support_download_by_kind(&snap, support_kind)
    }

    pub fn prepare_delete(&self, language_code: &str, feature: Feature) -> DeletePlan {
        let snap = self.snapshot();
        let code = LanguageCode::from(language_code);
        match feature {
            Feature::Core => {
                if let Some(engine) = BERGAMOT_ENGINE.get() {
                    engine
                        .lock()
                        .expect("engine lock poisoned")
                        .evict_involving(language_code);
                }
                #[cfg(feature = "dictionary")]
                self.close_dictionary(&snap, &code);
                #[cfg(feature = "tts")]
                self.clear_speech_cache();
                plan_delete_language(&snap, &code)
            }
            Feature::Dictionary => {
                #[cfg(feature = "dictionary")]
                self.close_dictionary(&snap, &code);
                plan_delete_dictionary(&snap, &code)
            }
            Feature::Tts => {
                #[cfg(feature = "tts")]
                self.clear_speech_cache();
                plan_delete_tts(&snap, &code)
            }
        }
    }

    pub fn prepare_delete_support_by_kind(&self, support_kind: &str) -> DeletePlan {
        let snap = self.snapshot();
        plan_delete_support_by_kind(&snap, support_kind)
    }

    #[cfg(feature = "dictionary")]
    fn close_dictionary(&self, snapshot: &CatalogSnapshot, language_code: &LanguageCode) {
        let mut cache = self.dictionaries.lock().expect("dictionary cache poisoned");
        close_dictionary_in_snapshot(snapshot, &mut cache, language_code);
    }

    pub fn prepare_delete_superseded_tts(
        &self,
        language_code: &str,
        selected_pack_id: &str,
    ) -> DeletePlan {
        let snap = self.snapshot();
        let code = LanguageCode::from(language_code);
        #[cfg(feature = "tts")]
        self.clear_speech_cache();
        plan_delete_superseded_tts(&snap, &code, selected_pack_id)
    }

    #[cfg(feature = "tts")]
    fn clear_speech_cache(&self) {
        self.speech.lock().expect("speech cache poisoned").clear();
    }

    pub fn size_bytes(&self, language_code: &str, feature: Feature) -> u64 {
        let snap = self.snapshot();
        let code = LanguageCode::from(language_code);
        match feature {
            Feature::Core => snap.catalog.translation_size_bytes_for_language(&code),
            Feature::Tts => snap.catalog.tts_size_bytes_for_language(&code),
            Feature::Dictionary => snap
                .catalog
                .language_by_code(&code)
                .and_then(|lang| {
                    snap.catalog
                        .dictionary_info(&crate::api::DictionaryCode::from(
                            lang.dictionary_code.clone(),
                        ))
                })
                .map(|info| info.size as u64)
                .unwrap_or(0),
        }
    }

    pub fn support_size_bytes_by_kind(&self, support_kind: &str) -> u64 {
        self.snapshot()
            .catalog
            .support_size_bytes_by_kind(support_kind)
    }

    #[cfg(feature = "dictionary")]
    pub fn lookup_dictionary(
        &self,
        language_code: &str,
        word: &str,
    ) -> Result<Option<WordWithTaggedEntries>, TranslatorError> {
        let snap = self.snapshot();
        let mut cache = self.dictionaries.lock().expect("dictionary cache poisoned");
        lookup_dictionary_in_snapshot(&snap, &mut cache, &LanguageCode::from(language_code), word)
    }

    #[cfg(feature = "tts")]
    pub fn available_tts_voices(
        &self,
        language_code: &str,
    ) -> Result<Vec<TtsVoiceOption>, TranslatorError> {
        let snap = self.snapshot();
        let mut cache = self.speech.lock().expect("speech cache poisoned");
        available_tts_voices_in_snapshot(&snap, &mut cache, &LanguageCode::from(language_code))
    }

    #[cfg(feature = "tts")]
    pub fn warm_tts_model(&self, language_code: &str) -> Result<(), TranslatorError> {
        let snap = self.snapshot();
        let mut cache = self.speech.lock().expect("speech cache poisoned");
        warm_tts_model_in_snapshot(&snap, &mut cache, &LanguageCode::from(language_code))
    }

    #[cfg(feature = "tts")]
    pub fn plan_speech_chunks(
        &self,
        language_code: &str,
        text: &str,
    ) -> Result<Vec<SpeechChunk>, TranslatorError> {
        let snap = self.snapshot();
        let mut cache = self.speech.lock().expect("speech cache poisoned");
        plan_speech_chunks_for_text_in_snapshot(
            &snap,
            &mut cache,
            &LanguageCode::from(language_code),
            text,
        )
    }

    #[cfg(feature = "tts")]
    pub fn synthesize_pcm(
        &self,
        language_code: &str,
        text: &str,
        speech_speed: f32,
        voice_name: Option<&str>,
        is_phonemes: bool,
    ) -> Result<PcmAudio, TranslatorError> {
        let snap = self.snapshot();
        let mut cache = self.speech.lock().expect("speech cache poisoned");
        synthesize_pcm_in_snapshot(
            &snap,
            &mut cache,
            &LanguageCode::from(language_code),
            text,
            speech_speed,
            voice_name.map(VoiceName::from).as_ref(),
            is_phonemes,
        )
    }

    pub fn transliterate(&self, text: &str, language_code: &str) -> Option<String> {
        #[cfg(feature = "transliterate")]
        {
            let snap = self.snapshot();
            let language = snap
                .catalog
                .language_by_code(&LanguageCode::from(language_code))?;
            let mucab_path = std::path::Path::new(&snap.base_dir)
                .join("bin")
                .join("mucab.bin");
            let mucab_path_str = mucab_path
                .exists()
                .then(|| mucab_path.to_string_lossy().into_owned());
            let source_script = crate::api::ScriptCode::from(language.script.clone());
            let target_script = crate::api::ScriptCode::from("Latn");
            crate::transliterate::transliterate_with_policy_for_language(
                text,
                &LanguageCode::from(language_code),
                &source_script,
                &target_script,
                mucab_path_str.as_deref(),
                true,
            )
        }
        #[cfg(not(feature = "transliterate"))]
        {
            let _ = (text, language_code);
            None
        }
    }
}

pub fn parse_selected_catalog(
    bundled_json: &str,
    disk_json: Option<&str>,
) -> Option<crate::catalog::LanguageCatalog> {
    let preferred = select_best_catalog(bundled_json, disk_json).ok()?;
    let fallback = if std::ptr::eq(preferred, bundled_json) {
        disk_json
    } else {
        Some(bundled_json)
    };

    parse_and_validate_catalog(preferred)
        .ok()
        .or_else(|| fallback.and_then(|json| parse_and_validate_catalog(json).ok()))
}

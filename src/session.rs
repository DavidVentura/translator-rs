use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::BergamotEngine;
use crate::catalog::{
    CatalogSnapshot, DeletePlan, DownloadPlan, FsPackInstallChecker, InstalledTtsPack,
    LanguageAvailabilityRow, LanguageOverview, PackInstallChecker, build_catalog_snapshot,
    build_language_overview, installed_tts_voices_for_language, language_rows_in_snapshot,
    parse_and_validate_catalog, plan_delete_dictionary, plan_delete_language,
    plan_delete_superseded_tts, plan_delete_support_by_kind, plan_delete_tts, plan_delete_tts_pack,
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

#[cfg(feature = "doc-align")]
use crate::doc_align::{DocAligner, DocumentDetection, DocumentQuad, WarpedImageRgba};
use crate::ocr::PreparedImageOverlay;
#[cfg(feature = "tesseract")]
use crate::ocr::ReadingOrder;
#[cfg(feature = "ppocr")]
use crate::ocr_runtime::translate_image_rgba_ppocr_in_snapshot;
#[cfg(feature = "tesseract")]
use crate::ocr_runtime::{OcrPool, translate_image_rgba_in_snapshot};
#[cfg(feature = "ppocr")]
use crate::ppocr::PpocrEngine;
#[cfg(feature = "tesseract")]
use crate::settings::PreferredOcrEngine;

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
    ocr: OcrPool,
    #[cfg(feature = "doc-align")]
    doc_align: Mutex<DocAlignCache>,
    #[cfg(feature = "ppocr")]
    ppocr: Mutex<PpocrCache>,
}

#[cfg(feature = "doc-align")]
struct DocAlignCache {
    state: Option<(String, Arc<DocAligner>)>,
}

#[cfg(feature = "doc-align")]
impl DocAlignCache {
    fn new() -> Self {
        Self { state: None }
    }
}

#[cfg(feature = "ppocr")]
struct PpocrCache {
    /// Cache key: (det path, rec path, keys path). Tied to the script — when the user
    /// switches source language, we may need to rebuild the recognizer for a different
    /// charset.
    state: Option<(String, String, String, Arc<PpocrEngine>)>,
}

#[cfg(feature = "ppocr")]
impl PpocrCache {
    fn new() -> Self {
        Self { state: None }
    }
}

/// Tesseract OCR worker pool size. A small fixed value: more workers
/// would multiply per-language tessdata RAM cost without much speedup,
/// since OCR scales sub-linearly past 4 cores on phone-class hardware.
#[cfg(feature = "tesseract")]
const OCR_POOL_SIZE: usize = 4;

impl TranslatorSession {
    pub fn from_snapshot(snapshot: CatalogSnapshot) -> Self {
        Self {
            snapshot: RwLock::new(Arc::new(snapshot)),
            #[cfg(feature = "tts")]
            speech: Mutex::new(SpeechCache::new()),
            #[cfg(feature = "dictionary")]
            dictionaries: Mutex::new(DictionaryCache::new()),
            #[cfg(feature = "tesseract")]
            ocr: OcrPool::new(OCR_POOL_SIZE),
            #[cfg(feature = "doc-align")]
            doc_align: Mutex::new(DocAlignCache::new()),
            #[cfg(feature = "ppocr")]
            ppocr: Mutex::new(PpocrCache::new()),
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
        max_image_size: u32,
        source_code: &str,
        target_code: &str,
        min_confidence: u32,
        reading_order: ReadingOrder,
        background_mode: BackgroundMode,
        preferred_engine: PreferredOcrEngine,
    ) -> Result<PreparedImageOverlay, TranslatorError> {
        let snap = self.snapshot();
        let src = LanguageCode::from(source_code);
        let tgt = LanguageCode::from(target_code);

        let (prepared_rgba, prepared_w, prepared_h) =
            prepare_image_for_ocr(rgba_bytes, width, height, max_image_size)?;
        let rgba_view: &[u8] = &prepared_rgba;

        #[cfg(feature = "ppocr")]
        if matches!(preferred_engine, PreferredOcrEngine::Paddle) {
            match self.ppocr_engine_for(&src) {
                Ok(ppocr) => {
                    log::info!("ocr engine: ppocr (source={})", source_code);
                    return translate_image_rgba_ppocr_in_snapshot(
                        self.engine(),
                        &ppocr,
                        &snap,
                        rgba_view,
                        prepared_w,
                        prepared_h,
                        &src,
                        &tgt,
                        background_mode,
                        reading_order,
                    )
                    .map_err(|e| {
                        if e.message.to_lowercase().contains("no text found") {
                            TranslatorError::ocr("No text found in image (engine=ppocr)")
                        } else {
                            e
                        }
                    });
                }
                Err(err) => log::warn!(
                    "ppocr unavailable for {}, falling back to tesseract: {err}",
                    source_code
                ),
            }
        }
        log::info!("ocr engine: tesseract (source={})", source_code);

        translate_image_rgba_in_snapshot(
            self.engine(),
            &self.ocr,
            &snap,
            rgba_view,
            prepared_w,
            prepared_h,
            &src,
            &tgt,
            min_confidence,
            reading_order,
            background_mode,
        )
        .map_err(|e| {
            if e.message.to_lowercase().contains("no text found") {
                TranslatorError::ocr("No text found in image (engine=tesseract)")
            } else {
                e
            }
        })
    }

    pub fn retranslate_prepared_overlay(
        &self,
        mut prepared: PreparedImageOverlay,
        source_code: &str,
        target_code: &str,
    ) -> Result<PreparedImageOverlay, TranslatorError> {
        let snap = self.snapshot();
        let src = LanguageCode::from(source_code);
        let tgt = LanguageCode::from(target_code);

        let block_texts: Vec<String> = prepared
            .blocks
            .iter()
            .map(|block| {
                block
                    .lines
                    .iter()
                    .map(|line| line.text.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        let non_empty_indices: Vec<usize> = block_texts
            .iter()
            .enumerate()
            .filter_map(|(index, text)| (!text.trim().is_empty()).then_some(index))
            .collect();

        if non_empty_indices.is_empty() {
            return Err(TranslatorError::ocr("No text found in image"));
        }

        let translated_blocks: Vec<String> = if src == tgt {
            block_texts.clone()
        } else {
            let texts_to_translate: Vec<String> = non_empty_indices
                .iter()
                .map(|&index| block_texts[index].clone())
                .collect();
            let translated = {
                let mut engine_guard = self.engine().lock().expect("bergamot engine lock poisoned");
                Translator::new(&mut engine_guard, &snap).translate_texts(
                    &src,
                    &tgt,
                    &texts_to_translate,
                )?
            };
            let mut merged = block_texts.clone();
            for (index, translated_text) in non_empty_indices
                .iter()
                .copied()
                .zip(translated.into_iter())
            {
                merged[index] = translated_text;
            }
            merged
        };

        for (block, translated_text) in prepared.blocks.iter_mut().zip(translated_blocks.iter()) {
            block.translated_text = translated_text.clone();
        }
        prepared.translated_text = translated_blocks.join("\n");

        Ok(prepared)
    }

    #[cfg(feature = "ppocr")]
    fn ppocr_engine_for(
        &self,
        source_code: &LanguageCode,
    ) -> Result<Arc<PpocrEngine>, TranslatorError> {
        use crate::api::TranslatorErrorKind;

        let snap = self.snapshot();
        let catalog = &snap.catalog;
        let rec_pack_id = catalog
            .ocr_pack_id_for_engine(source_code, "ppocr")
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::MissingAsset,
                    format!("no ppocr pack for language {}", source_code.as_str()),
                )
            })?;
        let rec_pack = catalog.pack(&rec_pack_id).ok_or_else(|| {
            TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                format!("ppocr pack {rec_pack_id} missing from catalog"),
            )
        })?;
        let det_pack_id = rec_pack.depends_on.first().ok_or_else(|| {
            TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                format!("ppocr pack {rec_pack_id} has no detector dependency"),
            )
        })?;
        let det_pack = catalog.pack(det_pack_id).ok_or_else(|| {
            TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                format!("ppocr detector pack {det_pack_id} missing from catalog"),
            )
        })?;

        let pack_installed = |pack_id: &str| {
            snap.pack_statuses
                .get(pack_id)
                .map(|s| s.installed)
                .unwrap_or(false)
        };
        if !pack_installed(&rec_pack_id) || !pack_installed(det_pack_id) {
            return Err(TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                format!(
                    "ppocr models not installed for {} (rec={}, det={})",
                    source_code.as_str(),
                    rec_pack_id,
                    det_pack_id
                ),
            ));
        }

        let base = std::path::Path::new(&snap.base_dir);
        let det_path = det_pack
            .files
            .iter()
            .find(|f| f.name.ends_with(".mnn"))
            .map(|f| base.join(&f.install_path))
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::MissingAsset,
                    "ppocr detector pack has no .mnn file",
                )
            })?;
        let rec_path = rec_pack
            .files
            .iter()
            .find(|f| f.name.ends_with(".mnn"))
            .map(|f| base.join(&f.install_path))
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::MissingAsset,
                    "ppocr rec pack has no .mnn file",
                )
            })?;
        let keys_path = rec_pack
            .files
            .iter()
            .find(|f| f.name.ends_with("_keys.txt"))
            .map(|f| base.join(&f.install_path))
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::MissingAsset,
                    "ppocr rec pack has no _keys.txt file",
                )
            })?;

        let det_str = det_path.to_string_lossy().into_owned();
        let rec_str = rec_path.to_string_lossy().into_owned();
        let keys_str = keys_path.to_string_lossy().into_owned();

        let mut cache = self.ppocr.lock().expect("ppocr cache poisoned");
        if let Some((d, r, k, engine)) = &cache.state {
            if d == &det_str && r == &rec_str && k == &keys_str {
                return Ok(Arc::clone(engine));
            }
        }
        let engine = Arc::new(PpocrEngine::load(&det_path, &rec_path, &keys_path, 4)?);
        cache.state = Some((det_str, rec_str, keys_str, Arc::clone(&engine)));
        Ok(engine)
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

    pub fn prepare_delete_tts_pack(&self, pack_id: &str) -> DeletePlan {
        let snap = self.snapshot();
        #[cfg(feature = "tts")]
        self.clear_speech_cache();
        plan_delete_tts_pack(&snap, pack_id)
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
    pub fn installed_tts_voices(&self, language_code: &str) -> Vec<InstalledTtsPack> {
        let snap = self.snapshot();
        installed_tts_voices_for_language(&snap, &LanguageCode::from(language_code))
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
        pack_id: Option<&str>,
    ) -> Result<Vec<SpeechChunk>, TranslatorError> {
        let snap = self.snapshot();
        let mut cache = self.speech.lock().expect("speech cache poisoned");
        plan_speech_chunks_for_text_in_snapshot(
            &snap,
            &mut cache,
            &LanguageCode::from(language_code),
            text,
            pack_id,
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
        pack_id: Option<&str>,
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
            pack_id,
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

    #[cfg(feature = "doc-align")]
    fn doc_aligner(&self) -> Result<Arc<DocAligner>, TranslatorError> {
        let snap = self.snapshot();
        let files = snap.catalog.support_files_by_kind("doc_detect");
        let file = files.first().ok_or_else(|| {
            TranslatorError::new(
                crate::api::TranslatorErrorKind::MissingAsset,
                "no doc_detect support pack in catalog",
            )
        })?;
        let abs_path = std::path::Path::new(&snap.base_dir).join(&file.install_path);
        let abs_str = abs_path.to_string_lossy().into_owned();
        if !abs_path.exists() {
            return Err(TranslatorError::new(
                crate::api::TranslatorErrorKind::MissingAsset,
                format!("doc-align model not installed at {}", abs_str),
            ));
        }
        let mut cache = self.doc_align.lock().expect("doc-align cache poisoned");
        if let Some((ref path, ref aligner)) = cache.state {
            if path == &abs_str {
                return Ok(Arc::clone(aligner));
            }
        }
        let aligner = Arc::new(DocAligner::load(&abs_path, 2)?);
        cache.state = Some((abs_str, Arc::clone(&aligner)));
        Ok(aligner)
    }

    #[cfg(feature = "doc-align")]
    pub fn detect_document_quad(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<DocumentDetection>, TranslatorError> {
        let Some(mut detection) = self.doc_aligner()?.detect(rgba, width, height)? else {
            return Ok(None);
        };
        let refined =
            crate::doc_align_refine::refine_quad_with_quality(rgba, width, height, &detection.quad);
        // Suppress the pre-fill when the model's quad doesn't trace real image edges. Better to
        // show "no document detected" than to hand the user a confidently-wrong quad to nudge.
        if refined.quality == crate::doc_align_refine::QuadQuality::Bad {
            return Ok(None);
        }
        detection.quad = refined.quad;
        Ok(Some(detection))
    }

    #[cfg(feature = "doc-align")]
    pub fn warp_document_rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        quad: &DocumentQuad,
        out_width: Option<u32>,
        out_height: Option<u32>,
        postprocess: bool,
    ) -> Result<WarpedImageRgba, TranslatorError> {
        let (default_w, default_h) = crate::doc_align::suggested_output_dims(quad);
        let out_w = out_width.unwrap_or(default_w);
        let out_h = out_height.unwrap_or(default_h);
        crate::doc_align::warp(rgba, width, height, quad, out_w, out_h, postprocess)
    }

    #[cfg(feature = "doc-align")]
    pub fn suggested_warp_dims(&self, quad: &DocumentQuad) -> (u32, u32) {
        crate::doc_align::suggested_output_dims(quad)
    }
}

/// Resize RGBA to `max_image_size` on the longest side (if larger) and apply CLAHE on the result.
/// Centralizes what was previously split between Kotlin's `ImageProcessor.downscaleImage` and
/// `doc_align::warp`'s post-process step. Single CLAHE pass at OCR-target resolution → less work
/// when input is larger than the OCR target, no double-CLAHE on doc-aligned inputs.
fn prepare_image_for_ocr(
    rgba: &[u8],
    width: u32,
    height: u32,
    max_image_size: u32,
) -> Result<(Vec<u8>, u32, u32), TranslatorError> {
    let t_total = std::time::Instant::now();
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| TranslatorError::ocr("image dims overflow"))?;
    if rgba.len() != expected {
        return Err(TranslatorError::ocr(format!(
            "rgba length {} != {}x{}x4 ({})",
            rgba.len(),
            width,
            height,
            expected
        )));
    }
    let longest = width.max(height);
    // Skip resize when the source is already within 5% of the cap. A full Triangle convolution
    // on millions of RGBA pixels costs ~75ms just to shrink by a couple percent; the OCR engines
    // tolerate that slack fine.
    let resize_threshold = (max_image_size as f32 * 1.05) as u32;
    let (mut out, out_w, out_h, resize_ms) = if longest <= resize_threshold {
        (rgba.to_vec(), width, height, 0.0_f32)
    } else {
        let t = std::time::Instant::now();
        let scale = max_image_size as f32 / longest as f32;
        let new_w = ((width as f32 * scale).round() as u32).max(1);
        let new_h = ((height as f32 * scale).round() as u32).max(1);
        let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
            .ok_or_else(|| TranslatorError::ocr("internal: failed to wrap rgba in RgbaImage"))?;
        let resized =
            image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);
        let resize_ms = t.elapsed().as_secs_f32() * 1000.0;
        (resized.into_raw(), new_w, new_h, resize_ms)
    };
    let t_clahe = std::time::Instant::now();
    #[cfg(feature = "doc-align")]
    crate::doc_align::apply_clahe(
        &mut out,
        out_w,
        out_h,
        crate::doc_align::CLAHE_CLIP_LIMIT,
        crate::doc_align::CLAHE_TILES,
        crate::doc_align::CLAHE_TILES,
    );
    let clahe_ms = t_clahe.elapsed().as_secs_f32() * 1000.0;
    log::info!(
        "prepare_image_for_ocr: {}x{} -> {}x{} (max={}) resize={:.1}ms clahe={:.1}ms total={:.1}ms",
        width,
        height,
        out_w,
        out_h,
        max_image_size,
        resize_ms,
        clahe_ms,
        t_total.elapsed().as_secs_f32() * 1000.0,
    );
    Ok((out, out_w, out_h))
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

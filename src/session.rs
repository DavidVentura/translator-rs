use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::{BergamotEngine, TranslateCtx};
#[cfg(feature = "ppocr")]
use crate::catalog::PackKind;
use crate::catalog::{
    CatalogSnapshot, DeletePlan, DownloadPlan, FsPackInstallChecker, LanguageAvailabilityRow,
    LanguageOverview, PackInstallChecker, build_catalog_snapshot, build_language_overview,
    language_rows_in_snapshot, parse_and_validate_catalog, plan_delete_dictionary,
    plan_delete_language, plan_delete_superseded_tts, plan_delete_support_by_kind, plan_delete_tts,
    plan_delete_tts_pack, plan_dictionary_download, plan_language_download,
    plan_support_download_by_kind, plan_tts_download, select_best_catalog,
};
use crate::routing::MixedTextTranslationResult;
use crate::settings::BackgroundMode;
use crate::styled::{OverlayScreenshot, StructuredTranslationResult, StyledFragment};
#[cfg(any(feature = "odt", feature = "epub"))]
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
#[cfg(any(feature = "tesseract", feature = "ppocr"))]
use crate::ocr::{OcrSourceSelection, ReadingOrder};
#[cfg(feature = "tesseract")]
use crate::ocr_runtime::{OcrPool, translate_image_rgba_in_snapshot};
#[cfg(feature = "ppocr")]
use crate::ocr_runtime::{
    ocr_source_for_lines, recognizer_script_for_language, route_ppocr_predictions,
    translate_image_rgba_ppocr_in_snapshot,
};
#[cfg(feature = "ppocr")]
use crate::ppocr::{PpocrEngine, PpocrRecognizerSpec};
#[cfg(any(feature = "tesseract", feature = "ppocr"))]
use crate::settings::PreferredOcrEngine;

#[cfg(feature = "tts")]
use crate::api::VoiceName;
#[cfg(feature = "tts")]
use crate::catalog::{InstalledTtsPack, installed_tts_voices_for_language};
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
    /// Set true by `cancel_ongoing_work()` (from any thread) to abort the
    /// in-flight document translation. slimt worker threads poll it and stop
    /// within ~one batch. Document translations reset it to false at entry.
    /// Translations are serialized through the global engine mutex, so a
    /// single per-session flag unambiguously targets the active document.
    document_cancel: AtomicBool,
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
    /// Cache key: (detector path, optional classifier path, sorted list of
    /// installed recognizer (script, model, keys) tuples). One engine per
    /// session is enough: detector + classifier load once, recognizers load
    /// lazily per script via internal `OnceLock`. The key invalidates when
    /// the installed pack set changes so a freshly-downloaded recognizer
    /// gets picked up.
    state: Option<(PpocrEngineKey, Arc<PpocrEngine>)>,
}

#[cfg(feature = "ppocr")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PpocrEngineKey {
    detector_path: String,
    classifier_path: Option<String>,
    textline_orientation_path: Option<String>,
    recognizers: Vec<(crate::catalog::PpocrScript, String, String)>,
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
            document_cancel: AtomicBool::new(false),
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

    /// Request cancellation of the in-flight document translation. Safe to call
    /// from any thread (e.g. a UI "cancel" tap) while a `*_ctx` translation is
    /// running on another thread — slimt workers observe it within ~one batch.
    pub fn cancel_ongoing_work(&self) {
        self.document_cancel.store(true, Ordering::Relaxed);
    }

    /// Clear the cancellation flag. Called once at the start of a document
    /// translation so a previous cancel does not abort the next one.
    pub fn begin_document_translation(&self) {
        self.document_cancel.store(false, Ordering::Relaxed);
    }

    /// Cancellable, progress-reporting [`Self::translate_texts`]. `on_progress`
    /// is invoked from slimt worker threads with `(sentences_done, total)`;
    /// cancellation is driven by [`Self::cancel_ongoing_work`].
    pub(crate) fn translate_texts_ctx(
        &self,
        from_code: &str,
        to_code: &str,
        texts: &[String],
        on_progress: &(dyn Fn(usize, usize) + Sync),
    ) -> Result<Vec<String>, TranslatorError> {
        let ctx = TranslateCtx {
            cancel: &self.document_cancel,
            on_progress,
        };
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_texts_ctx(
            &LanguageCode::from(from_code),
            &LanguageCode::from(to_code),
            texts,
            &ctx,
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

    /// Cancellable, progress-reporting [`Self::translate_structured_fragments_batch`].
    /// `on_progress` is invoked from slimt worker threads with
    /// `(sentences_done, sentences_total)`; cancellation via
    /// [`Self::cancel_ongoing_work`].
    pub fn translate_structured_fragments_batch_ctx(
        &self,
        pages: &[&[StyledFragment]],
        forced_source_code: Option<&str>,
        target_code: &str,
        available_language_codes: &[LanguageCode],
        background_mode: BackgroundMode,
        on_progress: &(dyn Fn(usize, usize) + Sync),
    ) -> Result<Vec<StructuredTranslationResult>, TranslatorError> {
        let ctx = TranslateCtx {
            cancel: &self.document_cancel,
            on_progress,
        };
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap).translate_structured_fragments_batch_ctx(
            pages,
            forced_source_code.map(LanguageCode::from).as_ref(),
            &LanguageCode::from(target_code),
            available_language_codes,
            background_mode,
            &ctx,
        )
    }

    /// Cancellable, progress-reporting alignment translation for documents.
    #[cfg(any(feature = "odt", feature = "epub"))]
    pub(crate) fn translate_texts_with_alignment_ctx(
        &self,
        from_code: &LanguageCode,
        to_code: &LanguageCode,
        texts: &[String],
        on_progress: &(dyn Fn(usize, usize) + Sync),
    ) -> Result<Option<Vec<TranslationWithAlignment>>, TranslatorError> {
        let ctx = TranslateCtx {
            cancel: &self.document_cancel,
            on_progress,
        };
        let snap = self.snapshot();
        let mut engine = self.engine().lock().expect("engine lock poisoned");
        Translator::new(&mut engine, &snap)
            .translate_texts_with_alignment_ctx(from_code, to_code, texts, &ctx)
    }

    #[cfg(any(feature = "tesseract", feature = "ppocr"))]
    pub fn translate_image_rgba(
        &self,
        rgba_bytes: &[u8],
        width: u32,
        height: u32,
        max_image_size: u32,
        source_selection: OcrSourceSelection,
        target_code: &str,
        min_confidence: u32,
        reading_order: Option<ReadingOrder>,
        background_mode: BackgroundMode,
        preferred_engine: PreferredOcrEngine,
    ) -> Result<PreparedImageOverlay, TranslatorError> {
        let snap = self.snapshot();
        let tgt = LanguageCode::from(target_code);

        #[cfg(feature = "ppocr")]
        if matches!(preferred_engine, PreferredOcrEngine::Paddle) {
            match self.ppocr_engine(&snap) {
                Ok(ppocr) => {
                    log::info!("ocr engine: ppocr ({:?})", source_selection);
                    return translate_image_rgba_ppocr_in_snapshot(
                        self.engine(),
                        &ppocr,
                        &snap,
                        rgba_bytes,
                        width,
                        height,
                        max_image_size,
                        &source_selection,
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
                Err(err) => {
                    #[cfg(feature = "tesseract")]
                    {
                        log::warn!("ppocr unavailable, falling back to tesseract: {err}");
                    }
                    #[cfg(not(feature = "tesseract"))]
                    {
                        return Err(err);
                    }
                }
            }
        }

        #[cfg(feature = "tesseract")]
        {
            let OcrSourceSelection::Specific { language_code } = &source_selection else {
                return Err(TranslatorError::missing_asset(
                    "auto-source OCR requires the ppocr engine",
                ));
            };
            log::info!("ocr engine: tesseract (source={})", language_code.as_str());
            let _ = max_image_size;

            // Tesseract has no pre-OCR detection geometry to vote on — the
            // page-segmentation mode must be chosen up front, so auto means
            // horizontal here.
            translate_image_rgba_in_snapshot(
                self.engine(),
                &self.ocr,
                &snap,
                rgba_bytes,
                width,
                height,
                language_code,
                &tgt,
                min_confidence,
                reading_order.unwrap_or(ReadingOrder::LeftToRight),
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
        #[cfg(not(feature = "tesseract"))]
        {
            let _ = (
                rgba_bytes,
                width,
                height,
                max_image_size,
                min_confidence,
                reading_order,
                background_mode,
                preferred_engine,
            );
            Err(TranslatorError::missing_asset(
                "requested OCR engine unavailable",
            ))
        }
    }

    /// Live-OCR detect: takes a pre-built `OrientedImage` (the live pipeline's
    /// FrameHandle holds one of these). Boxes returned are in detection-image
    /// coords; multiply by `oriented.det_to_full` (per-axis) to lift to the
    /// canonical box coord space.
    #[cfg(feature = "ppocr")]
    pub(crate) fn detect_text_in_oriented_image(
        &self,
        oriented: &crate::live_frame::OrientedImage,
    ) -> Result<Vec<crate::ocr::DetectedTextBox>, TranslatorError> {
        let snap = self.snapshot();
        let ppocr = self.ppocr_engine(&snap)?;
        let rgb_det = oriented
            .rgb_det
            .as_ref()
            .expect("detect path requires build_with_rgb");
        ppocr.detect_only_image(rgb_det, crate::ppocr::PpocrProfile::Live)
    }

    /// Estimate the scene's reading-direction quadrant from a set of
    /// detections against the supplied `OrientedImage`. Wraps
    /// [`crate::live_session::estimate_canonical_quadrant`] so callers
    /// don't have to extract `rgb` / build a matching gray themselves.
    /// Returns `None` when the orientation model isn't installed, no
    /// wide boxes qualified, or consensus didn't reach the gate.
    #[cfg(all(feature = "ppocr", feature = "planar-tracker"))]
    pub(crate) fn estimate_canonical_quadrant_in_oriented_image(
        &self,
        oriented: &crate::live_frame::OrientedImage,
        boxes: &[crate::ocr::DetectedTextBox],
    ) -> Result<Option<crate::coords::Quadrant>, TranslatorError> {
        let snap = self.snapshot();
        let ppocr = self.ppocr_engine(&snap)?;
        let rgb = oriented
            .rgb
            .as_ref()
            .expect("estimate path requires build_with_rgb");
        let gray_display = image::imageops::grayscale(&rgb.to_rgb8());
        // Boxes are canonical; `rgb` may be rec-res — scale to crop correctly.
        let scaled = oriented.rec_scaled_boxes(boxes);
        Ok(crate::live_session::estimate_canonical_quadrant(
            &ppocr,
            rgb,
            &gray_display,
            &scaled,
        ))
    }

    /// Rec-based orientation estimator: enumerates the four canonical
    /// quadrants on the K best sample detections via the script-specific
    /// recognizer, picks the canonical with the highest avg confidence.
    /// More accurate than the textline-ori path on out-of-distribution
    /// content (signage, large glyphs, non-Latin scripts) because rec's
    /// confidence drops sharply on wrong orientation while the binary
    /// classifier falls back to its class prior.
    ///
    /// Requires a known script — use this on the forced-source-language
    /// path. Auto-source callers should use
    /// `estimate_canonical_quadrant_in_oriented_image` instead.
    ///
    /// Returns the winning canonical plus the rec lines from the sample
    /// boxes at that canonical, so the downstream rec pass can skip
    /// recognising those boxes a second time.
    #[cfg(all(feature = "ppocr", feature = "planar-tracker"))]
    pub(crate) fn estimate_canonical_via_rec_in_oriented_image(
        &self,
        oriented: &crate::live_frame::OrientedImage,
        boxes: &[crate::ocr::DetectedTextBox],
        script: crate::PpocrScript,
    ) -> Result<Option<crate::coords::Quadrant>, TranslatorError> {
        let snap = self.snapshot();
        let ppocr = self.ppocr_engine(&snap)?;
        Ok(crate::live_session::estimate_canonical_via_rec(
            &ppocr, oriented, boxes, script,
        ))
    }

    /// Resolve a forced source language code (e.g. "en", "ja") to the
    /// PPOCR script it routes to via the installed catalog. Returns
    /// `None` when no ppocr recognizer is installed for the language.
    /// Used by the live acquire pipeline to pick between the rec-based
    /// and textline-ori-based orientation estimators.
    #[cfg(feature = "ppocr")]
    pub(crate) fn ppocr_script_for_language_code(
        &self,
        language_code: &str,
    ) -> Option<crate::PpocrScript> {
        let snap = self.snapshot();
        let code = crate::LanguageCode::from(language_code);
        crate::ocr_runtime::recognizer_script_for_language(&snap, &code).ok()
    }

    /// Live-OCR recognize: takes the same `OrientedImage` (full-resolution crop +
    /// pre-built grayscale) and caller-supplied boxes in *display-orient
    /// full-crop* coords (i.e. already scaled up from detection coords).
    /// `canonical_quadrant` is the anchor's stored orientation (None for
    /// still-image / first-acquire paths that have no anchor).
    #[cfg(feature = "ppocr")]
    pub(crate) fn recognize_in_oriented_image(
        &self,
        oriented: &crate::live_frame::OrientedImage,
        boxes: &[crate::ocr::DetectedTextBox],
        source_selection: OcrSourceSelection,
        canonical_quadrant: Option<crate::coords::Quadrant>,
    ) -> Result<Vec<crate::ocr::RecognizedTextLine>, TranslatorError> {
        use crate::ppocr::PpocrProfile;

        let snap = self.snapshot();
        let ppocr = self.ppocr_engine(&snap)?;
        let rgb = oriented
            .rgb
            .as_ref()
            .expect("recognize path requires build_with_rgb");
        // PPOCR needs a gray buffer in the **same orientation** as
        // `rgb` (display orient) so its per-box strip crops line up.
        // `oriented.gray` is sensor-orient now (the tracker's frame),
        // so we derive a display-orient gray on the fly from `rgb`.
        // Fires per-recognize call (acquire / refresh), not per-frame.
        let rgb8 = rgb.to_rgb8();
        let gray_display = image::imageops::grayscale(&rgb8);
        match source_selection {
            OcrSourceSelection::Auto => {
                let predictions = ppocr.classify_text_boxes_image(
                    rgb,
                    &gray_display,
                    boxes,
                    canonical_quadrant,
                )?;
                let scripts = route_ppocr_predictions(&ppocr, &predictions, boxes)?;
                let mut lines = ppocr.recognize_text_in_boxes_image(
                    rgb,
                    &gray_display,
                    boxes,
                    &scripts,
                    PpocrProfile::Live,
                    canonical_quadrant,
                )?;
                let source = ocr_source_for_lines(&snap, &lines, None);
                if let Some(code) = source {
                    let code = code.as_str().to_owned();
                    for line in &mut lines {
                        if !line.text.trim().is_empty() {
                            line.source_code = Some(code.clone());
                        }
                    }
                }
                Ok(lines)
            }
            OcrSourceSelection::Specific { language_code } => {
                let script = recognizer_script_for_language(&snap, &language_code)?;
                let scripts = vec![script; boxes.len()];
                ppocr.recognize_text_in_boxes_image(
                    rgb,
                    &gray_display,
                    boxes,
                    &scripts,
                    PpocrProfile::Live,
                    canonical_quadrant,
                )
            }
        }
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

    /// Build (or reuse) the session's PPOCR engine. One engine per session covers
    /// detect / classify / recognize for every installed recognizer pack; the engine
    /// loads each recognizer model lazily on first use. The cache key invalidates
    /// only when the installed pack set changes (e.g. after a download), so day-to-day
    /// switches between auto and forced source reuse the same loaded detector and
    /// recognizers.
    #[cfg(feature = "ppocr")]
    fn ppocr_engine(&self, snap: &CatalogSnapshot) -> Result<Arc<PpocrEngine>, TranslatorError> {
        use crate::api::TranslatorErrorKind;
        use crate::catalog::{FileRole, OcrPack, PackRecord};

        let catalog = &snap.catalog;
        let pack_installed = |pack_id: &str| {
            snap.pack_statuses
                .get(pack_id)
                .map(|s| s.installed)
                .unwrap_or(false)
        };

        let (det_pack_id, det_pack) = catalog
            .packs
            .iter()
            .find(|(_, pack)| matches!(&pack.kind, PackKind::Ocr(OcrPack::PpocrDetector)))
            .ok_or_else(|| {
                TranslatorError::new(
                    TranslatorErrorKind::MissingAsset,
                    "ppocr detector pack missing from catalog",
                )
            })?;
        if !pack_installed(det_pack_id) {
            return Err(TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                "ppocr detector pack is not installed",
            ));
        }

        let base = std::path::Path::new(&snap.base_dir);
        let best_present = |pack: &PackRecord, role: &str| {
            pack.role_alternatives(role)
                .into_iter()
                .map(|f| base.join(&f.install_path))
                .find(|path| path.exists())
        };
        let det_path = best_present(det_pack, FileRole::DETECTOR).ok_or_else(|| {
            TranslatorError::missing_asset("ppocr detector pack has no detector model on disk")
        })?;
        let classifier_path = best_present(det_pack, FileRole::SCRIPT_CLASSIFIER);
        let textline_orientation_path = best_present(det_pack, FileRole::TEXTLINE_ORIENTATION);

        let mut specs = Vec::new();
        for (pack_id, pack) in &catalog.packs {
            if !pack_installed(pack_id) {
                continue;
            }
            let PackKind::Ocr(OcrPack::PpocrRecognizer { script }) = &pack.kind else {
                continue;
            };
            // A recognizer's charset must match its model, so alternates are
            // paired by priority: take the best recognizer whose same-priority
            // keys file is also on disk. A half-downloaded upgrade then falls
            // back to the older complete pair instead of mixing generations.
            let pair = pack
                .role_alternatives(FileRole::RECOGNIZER)
                .into_iter()
                .find_map(|rec| {
                    let model_path = base.join(&rec.install_path);
                    if !model_path.exists() {
                        return None;
                    }
                    let keys_path = pack
                        .role_alternatives(FileRole::KEYS)
                        .into_iter()
                        .find(|keys| keys.priority == rec.priority)
                        .map(|keys| base.join(&keys.install_path))
                        .filter(|path| path.exists())?;
                    Some((model_path, keys_path))
                });
            let Some((model_path, keys_path)) = pair else {
                continue;
            };
            specs.push(PpocrRecognizerSpec {
                script: *script,
                model_path,
                keys_path,
            });
        }
        specs.sort_by_key(|s| s.script.as_slug());

        let key = PpocrEngineKey {
            detector_path: det_path.to_string_lossy().into_owned(),
            classifier_path: classifier_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            textline_orientation_path: textline_orientation_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            recognizers: specs
                .iter()
                .map(|s| {
                    (
                        s.script,
                        s.model_path.to_string_lossy().into_owned(),
                        s.keys_path.to_string_lossy().into_owned(),
                    )
                })
                .collect(),
        };

        let mut cache = self.ppocr.lock().expect("ppocr cache poisoned");
        if let Some((cached_key, engine)) = &cache.state {
            if cached_key == &key {
                return Ok(Arc::clone(engine));
            }
        }
        // Optional ink-matte model, shipped alongside det in the OCR model dir.
        let ink_path = det_path
            .parent()
            .map(|d| d.join("ink.mnn"))
            .filter(|p| p.exists());
        let engine = Arc::new(PpocrEngine::load(
            &det_path,
            classifier_path.as_deref(),
            textline_orientation_path.as_deref(),
            specs,
            4,
            ink_path.as_deref(),
        )?);
        cache.state = Some((key, Arc::clone(&engine)));
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

//! The warm PPOCR engine handle. Owns the lazily-loaded [`PpocrEngine`]
//! (detector + classifier + per-script recognizers, keyed off the installed
//! pack set) and the OCR operations that drive it. The catalog snapshot is
//! *not* owned here — it lives on the session and is passed in per call, so a
//! freshly-installed pack rebuilds the engine on the next snapshot.

use std::sync::{Arc, Mutex};

use image::DynamicImage;
use translator_core::api::{TranslatorError, TranslatorErrorKind};
use translator_core::catalog::{
    CatalogSnapshot, FileRole, OcrPack, PackKind, PackRecord, PpocrScript,
};
use translator_core::coords::Quadrant;
use translator_core::ocr::{DetectedTextBox, OcrSourceSelection, RecognizedTextLine};
use translator_raster::live_frame::OrientedImage;

use crate::ocr_runtime::{
    ocr_source_for_lines, recognizer_script_for_language, route_ppocr_predictions,
};
use crate::ppocr::{InkStrip, PpocrEngine, PpocrProfile, PpocrRecognizerSpec};

#[derive(Debug, Clone, PartialEq, Eq)]
struct PpocrEngineKey {
    detector_path: String,
    classifier_path: Option<String>,
    textline_orientation_path: Option<String>,
    ink_path: Option<String>,
    recognizers: Vec<(PpocrScript, String, String)>,
}

/// Warm OCR engine: holds the loaded [`PpocrEngine`] and rebuilds it lazily
/// when the installed pack set (the key) changes. Cheap to clone-call through.
pub struct OcrEngine {
    warm: Mutex<Option<(PpocrEngineKey, Arc<PpocrEngine>)>>,
}

impl Default for OcrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OcrEngine {
    pub fn new() -> Self {
        Self {
            warm: Mutex::new(None),
        }
    }

    /// Get-or-build the warm [`PpocrEngine`] for `snap`'s installed packs. The
    /// detector + classifier load once; recognizers load lazily per script. The
    /// key invalidates when the installed pack set changes.
    pub fn engine(&self, snap: &CatalogSnapshot) -> Result<Arc<PpocrEngine>, TranslatorError> {
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

        // Optional ink-matte model. Resolved via the detector pack's declared install
        // path (not det_path.parent() — det may be a different OCR version dir).
        let ink_path = best_present(det_pack, FileRole::INK);

        let key = PpocrEngineKey {
            detector_path: det_path.to_string_lossy().into_owned(),
            classifier_path: classifier_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            textline_orientation_path: textline_orientation_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            ink_path: ink_path.as_ref().map(|p| p.to_string_lossy().into_owned()),
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

        let mut cache = self.warm.lock().expect("ppocr cache poisoned");
        if let Some((cached_key, engine)) = cache.as_ref() {
            if cached_key == &key {
                return Ok(Arc::clone(engine));
            }
        }
        let engine = Arc::new(PpocrEngine::load(
            &det_path,
            classifier_path.as_deref(),
            textline_orientation_path.as_deref(),
            specs,
            4,
            ink_path.as_deref(),
        )?);
        *cache = Some((key, Arc::clone(&engine)));
        Ok(engine)
    }

    /// Live-OCR detect over a pre-built [`OrientedImage`]. Boxes returned are in
    /// detection-image coords; multiply by `oriented.det_to_full` to lift them
    /// to the canonical box coord space.
    pub fn detect_text_in_oriented_image(
        &self,
        snap: &CatalogSnapshot,
        oriented: &OrientedImage,
    ) -> Result<Vec<DetectedTextBox>, TranslatorError> {
        let ppocr = self.engine(snap)?;
        let rgb_det = oriented
            .rgb_det
            .as_ref()
            .expect("detect path requires build_with_rgb");
        ppocr.detect_only_image(rgb_det, PpocrProfile::Live)
    }

    /// Per-box ink strips for `boxes` against the oriented image's colour `rgb`,
    /// keeping each strip's matte (ch0) and bold channel (ch1). 1:1 with
    /// `boxes`; entries are `None` for boxes the model couldn't matte, and every
    /// entry is `None` when no ink model is installed.
    pub fn ink_strips(
        &self,
        snap: &CatalogSnapshot,
        rgb: &DynamicImage,
        boxes: &[DetectedTextBox],
        canonical_quadrant: Option<Quadrant>,
    ) -> Result<Vec<Option<InkStrip>>, TranslatorError> {
        let ppocr = self.engine(snap)?;
        Ok(ppocr.ink_strips(rgb, boxes, canonical_quadrant))
    }

    /// Live-OCR recognize over the same [`OrientedImage`] with caller-supplied
    /// boxes in display-orient full-crop coords. `canonical_quadrant` is the
    /// anchor's stored orientation (`None` for still / first-acquire paths).
    pub fn recognize_in_oriented_image(
        &self,
        snap: &CatalogSnapshot,
        oriented: &OrientedImage,
        boxes: &[DetectedTextBox],
        source_selection: OcrSourceSelection,
        canonical_quadrant: Option<Quadrant>,
    ) -> Result<Vec<RecognizedTextLine>, TranslatorError> {
        let ppocr = self.engine(snap)?;
        let rgb = oriented
            .rgb
            .as_ref()
            .expect("recognize path requires build_with_rgb");
        match source_selection {
            OcrSourceSelection::Auto => {
                let predictions =
                    ppocr.classify_text_boxes_image(rgb, boxes, canonical_quadrant)?;
                let scripts = route_ppocr_predictions(&ppocr, &predictions, boxes)?;
                let mut lines = ppocr.recognize_text_in_boxes_image(
                    rgb,
                    boxes,
                    &scripts,
                    PpocrProfile::Live,
                    canonical_quadrant,
                )?;
                let source = ocr_source_for_lines(snap, &lines, None);
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
                let script = recognizer_script_for_language(snap, &language_code)?;
                let scripts = vec![script; boxes.len()];
                ppocr.recognize_text_in_boxes_image(
                    rgb,
                    boxes,
                    &scripts,
                    PpocrProfile::Live,
                    canonical_quadrant,
                )
            }
        }
    }
}

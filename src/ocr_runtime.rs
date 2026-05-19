#[cfg(feature = "tesseract")]
use std::path::Path;
#[cfg(not(feature = "tesseract"))]
use std::sync::Mutex;
#[cfg(feature = "tesseract")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "tesseract")]
use std::sync::{Mutex, MutexGuard};

use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::BergamotEngine;
use crate::catalog::CatalogSnapshot;
#[cfg(feature = "ppocr")]
use crate::catalog::{OcrPack, PackKind, PpocrScript};
#[cfg(feature = "ppocr")]
use crate::language_detect::detect_language_robust_code;
#[cfg(feature = "ppocr")]
use crate::live_frame::OrientedImage;
#[cfg(feature = "ppocr")]
use crate::ocr::{DetectedTextBox, OcrSourceSelection, OrientedRect, RecognizedTextLine, TextLine};
#[cfg(feature = "tesseract")]
use crate::ocr::{DetectedWord, build_text_blocks};
use crate::ocr::{PreparedImageOverlay, ReadingOrder, Rect, TextBlock, prepare_overlay_image};
#[cfg(feature = "ppocr")]
use crate::ppocr::{PpocrEngine, PpocrProfile, PpocrScriptClass, PpocrScriptPrediction};
use crate::settings::BackgroundMode;
#[cfg(feature = "tesseract")]
use crate::tesseract::DetectedWord as TesseractDetectedWord;
#[cfg(feature = "tesseract")]
use crate::tesseract::{PageSegMode, TesseractWrapper};
use crate::translate::Translator;

#[cfg(feature = "tesseract")]
struct OcrEngineState {
    engine: TesseractWrapper,
    language_spec: String,
    reading_order: ReadingOrder,
    tessdata_path: String,
}

#[cfg(feature = "tesseract")]
pub struct OcrCache {
    state: Option<OcrEngineState>,
}

#[cfg(feature = "tesseract")]
impl OcrCache {
    pub fn new() -> Self {
        Self { state: None }
    }
}

#[cfg(feature = "tesseract")]
impl Default for OcrCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool of [`OcrCache`] instances, one [`Mutex`] per slot, so concurrent
/// callers can run Tesseract in parallel. Single-language workloads keep
/// each slot's tessdata loaded on first use; the cost is roughly N copies
/// of the language model in RAM (50–80 MB each for `eng`).
#[cfg(feature = "tesseract")]
pub struct OcrPool {
    workers: Vec<Mutex<OcrCache>>,
    /// Round-robin pointer used as a tiebreaker when every worker is busy
    /// — distributes the blocking-wait load instead of always queueing
    /// behind worker[0].
    next: AtomicUsize,
}

#[cfg(feature = "tesseract")]
impl OcrPool {
    pub fn new(n_workers: usize) -> Self {
        let n = n_workers.max(1);
        Self {
            workers: (0..n).map(|_| Mutex::new(OcrCache::new())).collect(),
            next: AtomicUsize::new(0),
        }
    }

    /// Lease an idle worker. Walks `try_lock` over every slot first
    /// (lock-free fast path); if all are busy, blocks on a round-robin
    /// pick so concurrent callers don't all queue behind the same slot.
    pub fn lease(&self) -> MutexGuard<'_, OcrCache> {
        for w in &self.workers {
            if let Ok(g) = w.try_lock() {
                return g;
            }
        }
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        self.workers[idx].lock().expect("ocr cache poisoned")
    }
}

#[cfg(feature = "tesseract")]
impl Default for OcrPool {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(feature = "tesseract")]
pub(crate) fn translate_image_rgba_in_snapshot(
    engine: &Mutex<BergamotEngine>,
    ocr_pool: &OcrPool,
    snapshot: &CatalogSnapshot,
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    source_code: &LanguageCode,
    target_code: &LanguageCode,
    min_confidence: u32,
    reading_order: ReadingOrder,
    background_mode: BackgroundMode,
) -> Result<PreparedImageOverlay, TranslatorError> {
    let blocks = build_tesseract_blocks(
        ocr_pool,
        snapshot,
        rgba_bytes,
        width,
        height,
        source_code,
        min_confidence,
        reading_order,
    )?;
    finalize_image_overlay(
        engine,
        snapshot,
        rgba_bytes,
        width,
        height,
        source_code,
        target_code,
        blocks,
        background_mode,
        reading_order,
    )
}

#[cfg(feature = "ppocr")]
pub(crate) fn translate_image_rgba_ppocr_in_snapshot(
    engine: &Mutex<BergamotEngine>,
    ppocr: &PpocrEngine,
    snapshot: &CatalogSnapshot,
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    max_image_size: u32,
    source_selection: &OcrSourceSelection,
    target_code: &LanguageCode,
    background_mode: BackgroundMode,
    reading_order: ReadingOrder,
) -> Result<PreparedImageOverlay, TranslatorError> {
    let full_rect = Rect {
        left: 0,
        top: 0,
        right: width,
        bottom: height,
    };
    let det_max_pixels = saturating_square(max_image_size);
    let oriented =
        OrientedImage::build_with_rgb(rgba_bytes, width, height, 0, full_rect, det_max_pixels)
            .map_err(|e| TranslatorError::ocr(format!("ppocr build oriented image failed: {e}")))?;
    let rgb = oriented.rgb.as_ref().expect("with_rgb path");
    let rgb_det = oriented.rgb_det.as_ref().expect("with_rgb path");
    // PPOCR needs display-orient gray (same coord frame as `rgb` and
    // detected boxes). `oriented.gray` is sensor-orient for the
    // tracker; we don't reuse it here.
    let rgb8 = rgb.to_rgb8();
    let gray_display = image::imageops::grayscale(&rgb8);

    let det_raw = ppocr
        .detect_only_image(rgb_det, PpocrProfile::Still)
        .map_err(|e| TranslatorError::ocr(format!("ppocr detection failed: {e}")))?;
    let det_boxes: Vec<DetectedTextBox> = det_raw
        .into_iter()
        .map(|b| scale_detected_box(b, oriented.det_to_full_scale, width, height))
        .collect();

    let scripts = match source_selection {
        OcrSourceSelection::Auto => {
            let predictions = ppocr
                .classify_text_boxes_image(rgb, &gray_display, &det_boxes)
                .map_err(|e| {
                    TranslatorError::ocr(format!("ppocr script classification failed: {e}"))
                })?;
            route_ppocr_predictions(ppocr, &predictions, &det_boxes)?
        }
        OcrSourceSelection::Specific { language_code } => {
            let script = recognizer_script_for_language(snapshot, language_code)?;
            vec![script; det_boxes.len()]
        }
    };

    let lines = ppocr
        .recognize_text_in_boxes_image(
            rgb,
            &gray_display,
            &det_boxes,
            &scripts,
            PpocrProfile::Still,
        )
        .map_err(|e| TranslatorError::ocr(format!("ppocr recognition failed: {e}")))?;

    let blocks = still_ppocr_lines_to_blocks(&det_boxes, lines);
    let source_code = match source_selection {
        OcrSourceSelection::Specific { language_code } => language_code.clone(),
        OcrSourceSelection::Auto => {
            let blocks_text = blocks
                .iter()
                .map(TextBlock::translation_text)
                .collect::<Vec<_>>()
                .join("\n");
            let Some(detected) = ocr_source_from_text(snapshot, &blocks_text, Some(target_code))
            else {
                return Err(TranslatorError::ocr(
                    "could not detect image source language",
                ));
            };
            detected
        }
    };
    finalize_image_overlay(
        engine,
        snapshot,
        rgba_bytes,
        width,
        height,
        &source_code,
        target_code,
        blocks,
        background_mode,
        reading_order,
    )
}

#[cfg(feature = "ppocr")]
fn saturating_square(side: u32) -> u32 {
    let n = (side as u64).saturating_mul(side as u64);
    n.min(u32::MAX as u64) as u32
}

#[cfg(feature = "ppocr")]
fn scale_detected_box(b: DetectedTextBox, scale: f32, max_w: u32, max_h: u32) -> DetectedTextBox {
    let left = ((b.rect.left as f32) * scale).max(0.0) as u32;
    let top = ((b.rect.top as f32) * scale).max(0.0) as u32;
    let right = ((b.rect.right as f32) * scale).min(max_w as f32) as u32;
    let bottom = ((b.rect.bottom as f32) * scale).min(max_h as f32) as u32;
    let rect = Rect {
        left: left.min(right.saturating_sub(1)),
        top: top.min(bottom.saturating_sub(1)),
        right: right.max(left + 1),
        bottom: bottom.max(top + 1),
    };
    let scale_oriented = |o: OrientedRect| OrientedRect {
        cx: o.cx * scale,
        cy: o.cy * scale,
        width: o.width * scale,
        height: o.height * scale,
        angle_radians: o.angle_radians,
    };
    let contour = b.contour.iter().map(|v| v * scale).collect();
    DetectedTextBox {
        rect,
        oriented_box: scale_oriented(b.oriented_box),
        tight_box: scale_oriented(b.tight_box),
        contour,
        score: b.score,
    }
}

#[cfg(feature = "tesseract")]
fn build_tesseract_blocks(
    ocr_pool: &OcrPool,
    snapshot: &CatalogSnapshot,
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    source_code: &LanguageCode,
    min_confidence: u32,
    reading_order: ReadingOrder,
) -> Result<Vec<TextBlock>, TranslatorError> {
    let bytes_per_pixel = 4i32;
    let i_width = width as i32;
    let i_height = height as i32;
    let bytes_per_line = i_width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| TranslatorError::ocr("image width overflow"))?;

    let page_seg_mode = match reading_order {
        ReadingOrder::LeftToRight => PageSegMode::PsmAutoOsd,
        ReadingOrder::TopToBottomLeftToRight => PageSegMode::PsmSingleBlockVertText,
    };

    let join_without_spaces = source_code.as_str() == "ja";
    let relax_single_char_confidence = reading_order == ReadingOrder::TopToBottomLeftToRight;

    let mut cache_guard = ocr_pool.lease();
    with_ocr_engine(
        &mut cache_guard,
        snapshot,
        source_code.as_str(),
        reading_order,
        |ocr| {
            ocr.set_page_seg_mode(page_seg_mode);
            ocr.set_frame(
                rgba_bytes,
                i_width,
                i_height,
                bytes_per_pixel,
                bytes_per_line,
            )
            .map_err(|err| format!("failed to set OCR frame: {err}"))?;
            let words = ocr
                .get_word_boxes()
                .map_err(|err| format!("failed to read OCR words: {err}"))?;
            let detected_words = words
                .into_iter()
                .map(map_tesseract_word)
                .collect::<Vec<_>>();
            Ok(build_text_blocks(
                &detected_words,
                min_confidence,
                join_without_spaces,
                relax_single_char_confidence,
            ))
        },
    )
    .map_err(TranslatorError::ocr)
}

/// Pair the still-path detector boxes with their recognised lines, carrying the
/// tight oriented rect forward into [`TextLine`] so paragraph grouping has the
/// glyph-tight metric (the `RecognizedTextLine` shape doesn't carry it on its
/// own).
#[cfg(feature = "ppocr")]
fn still_ppocr_lines_to_blocks(
    boxes: &[DetectedTextBox],
    lines: Vec<RecognizedTextLine>,
) -> Vec<TextBlock> {
    let text_lines: Vec<TextLine> = boxes
        .iter()
        .zip(lines.into_iter())
        .filter(|(_, line)| !line.text.trim().is_empty())
        .map(|(b, line)| TextLine {
            text: line.text,
            bounding_box: line.rect,
            oriented_box: line.oriented_box,
            tight_box: b.tight_box,
            word_rects: vec![line.rect],
        })
        .collect();
    crate::ocr::group_lines_into_paragraphs(text_lines, Default::default())
}

/// Map a PULC script class to the best installed PPOCR recognizer script. Tries the
/// specialist first (Eslav for Cyrillic), then a general fallback. Returns `None`
/// when nothing applicable is installed.
#[cfg(feature = "ppocr")]
fn ppocr_script_for_class(ppocr: &PpocrEngine, class: PpocrScriptClass) -> Option<PpocrScript> {
    let candidates: &[PpocrScript] = match class {
        PpocrScriptClass::Arabic => &[PpocrScript::Arabic],
        PpocrScriptClass::Chinese | PpocrScriptClass::Japanese => &[PpocrScript::Cj],
        PpocrScriptClass::Cyrillic => &[PpocrScript::Eslav, PpocrScript::Cyrillic],
        PpocrScriptClass::Devanagari => &[PpocrScript::Devanagari],
        // PULC has a Kannada class but PPOCR has no Kannada recognizer; falls through to
        // the dominant-pack fallback in `route_ppocr_predictions`.
        PpocrScriptClass::Kannada => &[],
        PpocrScriptClass::Korean => &[PpocrScript::Korean],
        PpocrScriptClass::Tamil => &[PpocrScript::Ta],
        PpocrScriptClass::Telugu => &[PpocrScript::Te],
        PpocrScriptClass::Latin => &[PpocrScript::Latin],
    };
    let installed: std::collections::HashSet<PpocrScript> = ppocr.installed_scripts().collect();
    candidates.iter().find(|s| installed.contains(s)).copied()
}

#[cfg(feature = "ppocr")]
pub(crate) fn recognizer_script_for_language(
    snapshot: &CatalogSnapshot,
    language_code: &LanguageCode,
) -> Result<PpocrScript, TranslatorError> {
    let pack_id = snapshot
        .catalog
        .ocr_pack_id_for_engine(language_code, "ppocr")
        .ok_or_else(|| {
            TranslatorError::missing_asset(format!(
                "no ppocr pack for language {}",
                language_code.as_str()
            ))
        })?;
    let pack = snapshot.catalog.pack(&pack_id).ok_or_else(|| {
        TranslatorError::missing_asset(format!("ppocr pack {pack_id} missing from catalog"))
    })?;
    match &pack.kind {
        PackKind::Ocr(OcrPack::PpocrRecognizer { script }) => Ok(*script),
        _ => Err(TranslatorError::missing_asset(format!(
            "pack {pack_id} is not a ppocr recognizer"
        ))),
    }
}

#[cfg(feature = "ppocr")]
const PPOCR_ROUTE_DOMINANT_MIN_RATIO: f32 = 0.55;
#[cfg(feature = "ppocr")]
const PPOCR_ROUTE_MINOR_KEEP_RATIO: f32 = 0.20;
#[cfg(feature = "ppocr")]
const PPOCR_ROUTE_SMOOTH_MIN_CLASSIFIED: usize = 8;

/// Resolve per-strip PULC predictions into per-strip PPOCR recognizer scripts. Strips
/// PULC could not classify fall back to the dominant classified script; minority
/// scripts below `PPOCR_ROUTE_MINOR_KEEP_RATIO` are folded into the dominant; Latin
/// strips in an otherwise single non-Latin batch fold into that non-Latin script
/// (PPOCR's non-Latin recognizers can handle Latin glyphs but not vice versa).
#[cfg(feature = "ppocr")]
pub(crate) fn route_ppocr_predictions(
    ppocr: &PpocrEngine,
    predictions: &[Option<PpocrScriptPrediction>],
    boxes: &[DetectedTextBox],
) -> Result<Vec<PpocrScript>, TranslatorError> {
    let latin_installed = ppocr
        .installed_scripts()
        .any(|s| s == PpocrScript::Latin)
        .then_some(PpocrScript::Latin);
    let mut routed: Vec<Option<PpocrScript>> = Vec::with_capacity(predictions.len());
    let mut classified = Vec::new();
    let mut missing_indices = Vec::new();
    for (idx, prediction) in predictions.iter().enumerate() {
        let Some(prediction) = prediction else {
            missing_indices.push(idx);
            routed.push(None);
            continue;
        };
        let script = ppocr_script_for_class(ppocr, prediction.class);
        if let Some(box_) = boxes.get(idx) {
            log::debug!(
                "ppocr route strip={} class={} score={:.3} det_score={:.3} width={} height={} area={} script={:?}",
                idx,
                prediction.class.name(),
                prediction.score,
                box_.score,
                box_.rect.width(),
                box_.rect.height(),
                box_.rect.width().saturating_mul(box_.rect.height()),
                script.as_ref().map(PpocrScript::as_slug),
            );
        }
        match script {
            Some(script) => {
                classified.push(script);
                routed.push(Some(script));
            }
            None => {
                missing_indices.push(idx);
                routed.push(None);
            }
        }
    }

    let fallback = dominant_script(&classified);
    if !missing_indices.is_empty() {
        let Some(fallback) = fallback else {
            return Err(TranslatorError::ocr(
                "could not classify script for any detected text strip",
            ));
        };
        for idx in missing_indices {
            log::debug!(
                "ppocr route strip={} script=unknown -> {} (fallback=dominant)",
                idx,
                fallback.as_slug(),
            );
            routed[idx] = Some(fallback);
        }
    }

    smooth_dominant_routes(&mut routed, &classified);

    if let Some(latin) = latin_installed {
        let non_latin: std::collections::HashSet<PpocrScript> = routed
            .iter()
            .filter_map(|s| *s)
            .filter(|s| *s != latin)
            .collect();
        if non_latin.len() == 1 {
            let target = *non_latin.iter().next().unwrap();
            for slot in &mut routed {
                if *slot == Some(latin) {
                    *slot = Some(target);
                }
            }
            log::debug!(
                "ppocr route merged latin into {} for mixed-script batch",
                target.as_slug(),
            );
        }
    }

    Ok(routed
        .into_iter()
        .map(|s| s.expect("all ppocr routes populated"))
        .collect())
}

#[cfg(feature = "ppocr")]
fn dominant_script(scripts: &[PpocrScript]) -> Option<PpocrScript> {
    scripts
        .iter()
        .fold(
            std::collections::HashMap::<PpocrScript, usize>::new(),
            |mut counts, script| {
                *counts.entry(*script).or_default() += 1;
                counts
            },
        )
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(script, _)| script)
}

#[cfg(feature = "ppocr")]
fn smooth_dominant_routes(routed: &mut [Option<PpocrScript>], classified: &[PpocrScript]) {
    if classified.len() < PPOCR_ROUTE_SMOOTH_MIN_CLASSIFIED {
        return;
    }
    let counts: std::collections::HashMap<PpocrScript, usize> =
        classified
            .iter()
            .fold(std::collections::HashMap::new(), |mut counts, script| {
                *counts.entry(*script).or_default() += 1;
                counts
            });
    let Some((&dominant, &dominant_count)) = counts.iter().max_by_key(|(_, count)| **count) else {
        return;
    };
    let total = classified.len() as f32;
    let dominant_ratio = dominant_count as f32 / total;
    if dominant_ratio < PPOCR_ROUTE_DOMINANT_MIN_RATIO {
        return;
    }

    let minority: std::collections::HashSet<PpocrScript> = counts
        .iter()
        .filter_map(|(&script, &count)| {
            let ratio = count as f32 / total;
            (script != dominant && ratio < PPOCR_ROUTE_MINOR_KEEP_RATIO).then_some(script)
        })
        .collect();
    for script in &minority {
        log::debug!(
            "ppocr route smoothing: folding script={} into dominant={} dominant_ratio={:.2}",
            script.as_slug(),
            dominant.as_slug(),
            dominant_ratio,
        );
    }
    for slot in routed.iter_mut() {
        if let Some(s) = slot {
            if minority.contains(s) {
                *slot = Some(dominant);
            }
        }
    }
}

/// Run CLD over recognized text and pick the best installed source language. When
/// `target_code` is given, the picked language must also be translatable to that
/// target (otherwise returns `None`). Used both still-mode (where the caller turns
/// `None` into a `MissingAsset`) and live-mode (where the caller keeps `None` so the
/// frame renders as untranslated text).
#[cfg(feature = "ppocr")]
fn ocr_source_from_text(
    snapshot: &CatalogSnapshot,
    text: &str,
    target_code: Option<&LanguageCode>,
) -> Option<LanguageCode> {
    let available = snapshot
        .availability_by_code
        .keys()
        .map(|code| LanguageCode::from(code.as_str()))
        .collect::<Vec<_>>();
    let detected = detect_language_robust_code(text, None, &available)?;
    if let Some(target) = target_code {
        if !snapshot.can_translate(&detected, target) {
            return None;
        }
    }
    Some(detected)
}

#[cfg(feature = "ppocr")]
pub(crate) fn ocr_source_for_lines(
    snapshot: &CatalogSnapshot,
    lines: &[RecognizedTextLine],
    target_code: Option<&LanguageCode>,
) -> Option<LanguageCode> {
    let text = lines
        .iter()
        .map(|line| line.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    ocr_source_from_text(snapshot, &text, target_code)
}

fn finalize_image_overlay(
    engine: &Mutex<BergamotEngine>,
    snapshot: &CatalogSnapshot,
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    source_code: &LanguageCode,
    target_code: &LanguageCode,
    blocks: Vec<TextBlock>,
    background_mode: BackgroundMode,
    reading_order: ReadingOrder,
) -> Result<PreparedImageOverlay, TranslatorError> {
    let t_translate = std::time::Instant::now();
    let translated_blocks = {
        let mut engine_guard = engine.lock().expect("bergamot engine lock poisoned");
        translate_block_texts(
            &mut engine_guard,
            snapshot,
            source_code,
            target_code,
            &blocks,
        )?
    };
    let translate_ms = t_translate.elapsed().as_secs_f32() * 1000.0;

    let t_overlay = std::time::Instant::now();
    let result = prepare_overlay_image(
        rgba_bytes,
        width,
        height,
        &blocks,
        &translated_blocks,
        background_mode,
        reading_order,
    )
    .map_err(TranslatorError::ocr);
    let overlay_ms = t_overlay.elapsed().as_secs_f32() * 1000.0;
    log::info!(
        "finalize_image_overlay: {} blocks — translate={:.1}ms overlay_prep={:.1}ms",
        blocks.len(),
        translate_ms,
        overlay_ms,
    );
    result
}

#[cfg(feature = "tesseract")]
fn map_tesseract_word(word: TesseractDetectedWord) -> DetectedWord {
    DetectedWord {
        text: word.text,
        confidence: word.confidence,
        bounding_box: Rect {
            left: word.bounding_rect.left as u32,
            top: word.bounding_rect.top as u32,
            right: word.bounding_rect.right as u32,
            bottom: word.bounding_rect.bottom as u32,
        },
        is_at_beginning_of_para: word.is_at_beginning_of_para,
        end_para: word.end_para,
        end_line: word.end_line,
    }
}

fn translate_block_texts(
    engine: &mut BergamotEngine,
    snapshot: &CatalogSnapshot,
    source_code: &LanguageCode,
    target_code: &LanguageCode,
    blocks: &[TextBlock],
) -> Result<Vec<String>, TranslatorError> {
    let block_texts = blocks
        .iter()
        .map(TextBlock::translation_text)
        .collect::<Vec<_>>();
    let non_empty_indices = block_texts
        .iter()
        .enumerate()
        .filter_map(|(index, text)| (!text.trim().is_empty()).then_some(index))
        .collect::<Vec<_>>();

    if non_empty_indices.is_empty() {
        return Err(TranslatorError::ocr("No text found in image"));
    }

    if source_code == target_code {
        return Ok(block_texts);
    }

    let texts_to_translate = non_empty_indices
        .iter()
        .map(|&index| block_texts[index].clone())
        .collect::<Vec<_>>();
    let translated = Translator::new(engine, snapshot).translate_texts(
        source_code,
        target_code,
        &texts_to_translate,
    )?;

    Ok(merge_translated_block_texts(
        &block_texts,
        &non_empty_indices,
        translated,
    ))
}

fn merge_translated_block_texts(
    block_texts: &[String],
    non_empty_indices: &[usize],
    translated_non_empty: Vec<String>,
) -> Vec<String> {
    let mut translated_blocks = block_texts.to_vec();
    for (index, translated_text) in non_empty_indices
        .iter()
        .copied()
        .zip(translated_non_empty.into_iter())
    {
        translated_blocks[index] = translated_text;
    }
    translated_blocks
}

#[cfg(feature = "tesseract")]
fn with_ocr_engine<T, F>(
    cache: &mut OcrCache,
    snapshot: &CatalogSnapshot,
    source_code: &str,
    reading_order: ReadingOrder,
    f: F,
) -> Result<T, String>
where
    F: FnOnce(&mut TesseractWrapper) -> Result<T, String>,
{
    let language = snapshot
        .catalog
        .language_by_code(&LanguageCode::from(source_code))
        .ok_or_else(|| format!("unknown source language: {source_code}"))?;
    let tessdata_path = Path::new(&snapshot.base_dir)
        .join("tesseract")
        .join("tessdata");
    let has_japanese_vertical_model =
        source_code == "ja" && tessdata_path.join("jpn_vert.traineddata").exists();
    let language_spec = match (source_code, reading_order, has_japanese_vertical_model) {
        ("ja", ReadingOrder::TopToBottomLeftToRight, true) => "jpn_vert".to_string(),
        _ => format!("{}+eng", language.tess_name),
    };

    let tessdata_path_string = tessdata_path.to_string_lossy().into_owned();
    let needs_reinit = cache.state.as_ref().is_none_or(|state| {
        state.language_spec != language_spec
            || state.reading_order != reading_order
            || state.tessdata_path != tessdata_path_string
    });

    if needs_reinit {
        let engine = TesseractWrapper::new(
            Some(
                tessdata_path
                    .to_str()
                    .ok_or_else(|| "invalid tessdata path".to_string())?,
            ),
            Some(&language_spec),
        )
        .map_err(|err| format!("failed to initialize tesseract: {err}"))?;
        cache.state = Some(OcrEngineState {
            engine,
            language_spec,
            reading_order,
            tessdata_path: tessdata_path_string,
        });
    }

    let state = cache
        .state
        .as_mut()
        .ok_or_else(|| "OCR engine unavailable".to_string())?;
    f(&mut state.engine)
}

#[cfg(test)]
mod tests {
    use super::merge_translated_block_texts;

    #[test]
    fn preserves_blank_blocks_when_merging_translations() {
        let block_texts = vec![
            "hello".to_string(),
            String::new(),
            "world".to_string(),
            "   ".to_string(),
        ];

        let merged = merge_translated_block_texts(
            &block_texts,
            &[0, 2],
            vec!["hola".to_string(), "mundo".to_string()],
        );

        assert_eq!(
            merged,
            vec![
                "hola".to_string(),
                String::new(),
                "mundo".to_string(),
                "   ".to_string(),
            ]
        );
    }
}

use std::sync::Mutex;

use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::BergamotEngine;
use crate::catalog::CatalogSnapshot;
use crate::catalog::{OcrPack, PackKind, PpocrScript};
use crate::language_detect::detect_language_robust_code;
use crate::live_frame::OrientedImage;
use crate::ocr::{DetectedTextBox, OcrSourceSelection, OrientedRect, RecognizedTextLine, TextLine};
use crate::ocr::{PreparedImageOverlay, ReadingOrder, Rect, TextBlock, prepare_overlay_image};
use crate::ppocr::{PpocrEngine, PpocrProfile, PpocrScriptClass, PpocrScriptPrediction};
use crate::settings::BackgroundMode;
use crate::text_metrics::{LineMetrics, measure_line};
use crate::translate::Translator;

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
    min_confidence: u32,
    background_mode: BackgroundMode,
    reading_order: Option<ReadingOrder>,
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
        .map(|b| scale_detected_box(b, oriented.det_to_full.0, width, height))
        .collect();

    // Still images are display-oriented, so the canonical reading frame is R0.
    // Passing it explicitly (instead of None) pins the dewarp direction of
    // near-vertical strips — CJK vertical columns — to top-char-first; with
    // None the PCA sign for those strips is per-column noise.
    let scripts = match source_selection {
        OcrSourceSelection::Auto => {
            let predictions = ppocr
                .classify_text_boxes_image(
                    rgb,
                    &gray_display,
                    &det_boxes,
                    Some(crate::coords::Quadrant::R0),
                )
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

    // No requested reading order: infer it. Only CJK pages can read
    // vertically (modern Korean is horizontal, so the gate is the shared
    // Chinese+Japanese recognizer script, not "any CJK language"), and within
    // those the detections' own long-axis orientations carry the answer.
    let reading_order = reading_order.unwrap_or_else(|| {
        let cj_boxes = scripts.iter().filter(|s| **s == PpocrScript::Cj).count();
        let cjk_dominant = cj_boxes * 2 >= scripts.len() && !scripts.is_empty();
        let resolved = if cjk_dominant && crate::ocr::detected_lines_read_vertically(&det_boxes) {
            ReadingOrder::TopToBottomRightToLeft
        } else {
            ReadingOrder::LeftToRight
        };
        log::info!(
            "ppocr auto reading order: {} cj boxes of {} → {:?}",
            cj_boxes,
            scripts.len(),
            resolved,
        );
        resolved
    });

    let lines = ppocr
        .recognize_text_in_boxes_image(
            rgb,
            &gray_display,
            &det_boxes,
            &scripts,
            PpocrProfile::Still,
            Some(crate::coords::Quadrant::R0),
        )
        .map_err(|e| TranslatorError::ocr(format!("ppocr recognition failed: {e}")))?;

    // Per-box ink mattes feed both paragraph grouping (x-height + baseline-tilt
    // recovery, applied to each line before it groups) and the overlay erase
    // (the union ink mask). Compute them once, here, before grouping needs them.
    let (ink_masks, ink_rgba) = if ppocr.has_ink() {
        let rgba = image::RgbaImage::from_raw(width, height, rgba_bytes.to_vec())
            .expect("rgba image from caller-owned bytes");
        let dynimg = image::DynamicImage::ImageRgba8(rgba);
        let masks = ppocr.ink_masks(&dynimg, &det_boxes);
        (masks, Some(dynimg.into_rgba8()))
    } else {
        (Vec::new(), None)
    };
    let text_metrics = box_line_metrics(&det_boxes, &ink_masks);

    let blocks = still_ppocr_lines_to_blocks(
        &det_boxes,
        lines,
        &text_metrics,
        min_confidence,
        reading_order,
    );
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

    // The same per-box mattes → one full-image union mask. The overlay erase
    // replaces just the inked pixels with a reconstructed background instead of
    // flat-filling each line's rect. `None` when no ink model is installed →
    // flat-fill fallback.
    let ink_union = ink_rgba
        .as_ref()
        .map(|rgba| crate::color_matting::union_ink_mask(rgba, &det_boxes, &ink_masks));

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
        ink_union.as_deref(),
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

/// Per-box ink-matte typography (x-height + baseline tilt), 1:1 with `boxes`.
/// `None` for a box with no matte (no ink model, degenerate box, or no coherent
/// ink band); the caller then keeps the box's own tight height and angle.
#[cfg(feature = "ppocr")]
fn box_line_metrics(
    boxes: &[DetectedTextBox],
    masks: &[Option<image::GrayImage>],
) -> Vec<Option<LineMetrics>> {
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mask = masks.get(i)?.as_ref()?;
            measure_line(mask, b.oriented_box.width, b.oriented_box.height)
        })
        .collect()
}

/// Pair the still-path detector boxes with their recognised lines, carrying the
/// tight oriented rect forward into [`TextLine`] so paragraph grouping has the
/// glyph-tight metric (the `RecognizedTextLine` shape doesn't carry it on its
/// own). Where the ink matte resolved a line's typography, rebuild that tight box
/// from the matte band — its height becomes the glyph-content-stable x-height, its
/// centre snaps to the band centreline, and its angle is corrected by the recovered
/// baseline tilt — so grouping keys size *and* spacing on the real ink, not the
/// detection box. The render `oriented_box` gets the same tilt correction.
#[cfg(feature = "ppocr")]
fn still_ppocr_lines_to_blocks(
    boxes: &[DetectedTextBox],
    lines: Vec<RecognizedTextLine>,
    text_metrics: &[Option<LineMetrics>],
    min_confidence: u32,
    reading_order: ReadingOrder,
) -> Vec<TextBlock> {
    // The user's confidence setting (0–100) is a stricter line gate on top of the
    // recognizer's built-in `rec_drop_score`: lines the model accepted but whose mean
    // CTC score sits below the user's bar are dropped before paragraph grouping.
    let min_score = min_confidence as f32 / 100.0;
    let text_lines: Vec<TextLine> = boxes
        .iter()
        .zip(lines.into_iter())
        .zip(text_metrics.iter())
        .filter(|((_, line), _)| !line.text.trim().is_empty() && line.confidence >= min_score)
        .map(|((b, line), metrics)| {
            let delta = metrics.map_or(0.0, |m| m.baseline_angle_delta);
            let mut oriented_box = line.oriented_box;
            oriented_box.angle_radians += delta;

            // Re-fit the grouping box to the actual ink (x-height, ink width,
            // centred on the ink) where the matte resolved it; otherwise keep the
            // detection box. The inflated `oriented_box` (render/erase footprint)
            // only takes the tilt correction.
            let tight_box = metrics.map_or(b.tight_box, |m| m.refit(b.tight_box));
            TextLine {
                text: line.text,
                bounding_box: line.rect,
                oriented_box,
                tight_box,
                word_rects: vec![line.rect],
            }
        })
        .collect();
    match reading_order {
        ReadingOrder::LeftToRight => {
            crate::ocr::group_lines_into_paragraphs(text_lines, Default::default())
        }
        ReadingOrder::TopToBottomRightToLeft => {
            crate::ocr::group_vertical_lines_into_paragraphs(text_lines, Default::default())
        }
    }
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
        // PULC's Kannada class is the only one of the merged Indic scripts (bn/gu/kn/ml)
        // it can name; the other three have no PULC class and reach the Indic recognizer
        // only via forced-source or the dominant-pack fallback.
        PpocrScriptClass::Kannada => &[PpocrScript::Indic],
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
    ink_mask: Option<&[bool]>,
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
        ink_mask,
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

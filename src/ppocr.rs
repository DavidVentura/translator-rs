use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use image::{DynamicImage, GenericImageView, GrayImage, RgbImage, imageops::FilterType};
use imageproc::contours::find_contours;
use imageproc::point::Point;
use rayon::ThreadPool;
use rayon::prelude::*;

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::catalog::PpocrScript;
use crate::mnn_inference::MnnSession;
use mnn_sys::{MemoryMode, PrecisionMode};

const REC_TARGET_HEIGHT: u32 = 48;
const REC_MIN_SCORE: f32 = 0.3;
/// Per-character CTC score gate for punctuation glyphs. Kept lower than `REC_MIN_SCORE`
/// because real punctuation is small and ambiguous (period vs comma vs apostrophe), so its
/// max-prob legitimately runs lower than letter glyphs. 0.3 still rejects the single rogue
/// glyphs that texture/JPEG-ringing artifacts produce on non-text regions.
const REC_PUNCT_MIN_SCORE: f32 = 0.3;
/// Line-level confidence gate — analogous to PaddleOCR upstream's `drop_score = 0.5`. Lines
/// whose mean accepted-character CTC score is below this threshold are discarded after
/// recognition. The dominant filter for spurious detections from non-text image regions (the
/// per-pixel `DET_SCORE_THRESHOLD` and per-character `REC_MIN_SCORE` gates accept
/// individually-weak signals; this rejects whole lines that never look strong on average).
const REC_DROP_SCORE: f32 = 0.5;
/// One rec session per parallel worker, dispatched via rayon. Mirrors the demo's strategy:
/// each session runs single-threaded (intra=1) on its own crop. Inter-session parallelism beats
/// intra-session padding-and-batching on this workload because crop widths are heterogeneous
/// and the per-graph compute is small enough that ORT/MNN intra-threading scales poorly.
const REC_PARALLELISM: usize = 4;
// Hard OOM ceiling, not a perf cap. The user's `maxImageSize` setting (and
// the doc-align warp step) already determine the working resolution; we only
// step in if something pathological reaches us.
const DET_MAX_SIDE: u32 = 4096;
const DET_SCORE_THRESHOLD: f32 = 0.3;
/// Mean DB-heatmap probability inside a contour's AABB before that contour is allowed to
/// become a detection. Analogous to upstream PaddleOCR's `det_db_box_thresh` (default 0.6) in
/// "fast" `score_mode`. Real text masks score 0.7+ on average inside their box; spurious
/// blobs from texture or compression noise typically sit at 0.35–0.45 — just enough to clear
/// the per-pixel `DET_SCORE_THRESHOLD` but well below this gate.
const DET_BOX_MIN_SCORE: f32 = 0.6;
/// Minimum contour AABB area (in detector-output mask pixels) for a detection to survive.
/// Tighter than upstream's effective minimum so noise blobs from texture / JPEG ringing /
/// foliage that just clear `DET_SCORE_THRESHOLD` get dropped before they ever hit the
/// recognizer. Real text glyphs at typical OCR scale fill many hundreds of mask pixels, so
/// the floor is comfortably below legitimate text.
const DET_MIN_AREA: u32 = 64;
const DET_UNCLIP_RATIO: f32 = 1.6;
const DET_BOX_BORDER: u32 = 4;
const LIVE_REC_DROP_SCORE: f32 = 0.65;
const LIVE_DET_BOX_MIN_SCORE: f32 = 0.68;
const LIVE_DET_MIN_AREA: u32 = 350;

const PPOCR_DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PPOCR_DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const PPOCR_REC_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const PPOCR_REC_STD: [f32; 3] = [0.5, 0.5, 0.5];
const PULC_WIDTH: u32 = 160;
const PULC_HEIGHT: u32 = 80;
const PULC_MIN_SCORE: f32 = 0.85;
const PULC_MIN_STRIP_AREA: u32 = 768;
const PULC_MIN_STRIP_WIDTH: u32 = 24;
const PULC_MIN_STRIP_HEIGHT: u32 = 8;
const PULC_MIN_IMAGE_AREA_RATIO: f32 = 0.00030;
/// PULC class index order — matches the model's softmax output layout. PULC's `chinese_cht`
/// class is trained on both simplified and traditional Chinese, so we name it `Chinese`.
const PULC_CLASSES: [PpocrScriptClass; 10] = [
    PpocrScriptClass::Arabic,
    PpocrScriptClass::Chinese,
    PpocrScriptClass::Cyrillic,
    PpocrScriptClass::Devanagari,
    PpocrScriptClass::Japanese,
    PpocrScriptClass::Kannada,
    PpocrScriptClass::Korean,
    PpocrScriptClass::Tamil,
    PpocrScriptClass::Telugu,
    PpocrScriptClass::Latin,
];

#[derive(Debug, Clone, Copy)]
struct PpocrThresholds {
    det_score_threshold: f32,
    det_box_min_score: f32,
    det_min_area: u32,
    rec_drop_score: f32,
}

const STILL_THRESHOLDS: PpocrThresholds = PpocrThresholds {
    det_score_threshold: DET_SCORE_THRESHOLD,
    det_box_min_score: DET_BOX_MIN_SCORE,
    det_min_area: DET_MIN_AREA,
    rec_drop_score: REC_DROP_SCORE,
};

const LIVE_THRESHOLDS: PpocrThresholds = PpocrThresholds {
    det_score_threshold: DET_SCORE_THRESHOLD,
    det_box_min_score: LIVE_DET_BOX_MIN_SCORE,
    det_min_area: LIVE_DET_MIN_AREA,
    rec_drop_score: LIVE_REC_DROP_SCORE,
};

/// Detection / recognition profile. `Live` uses stricter thresholds so transient texture
/// noise from a moving camera does not become a stable tracked overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpocrProfile {
    Still,
    Live,
}

impl PpocrProfile {
    fn thresholds(self) -> PpocrThresholds {
        match self {
            PpocrProfile::Still => STILL_THRESHOLDS,
            PpocrProfile::Live => LIVE_THRESHOLDS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PpocrScriptClass {
    Arabic,
    Chinese,
    Cyrillic,
    Devanagari,
    Japanese,
    Kannada,
    Korean,
    Tamil,
    Telugu,
    Latin,
}

impl PpocrScriptClass {
    pub fn name(&self) -> &'static str {
        match self {
            PpocrScriptClass::Arabic => "arabic",
            PpocrScriptClass::Chinese => "chinese",
            PpocrScriptClass::Cyrillic => "cyrillic",
            PpocrScriptClass::Devanagari => "devanagari",
            PpocrScriptClass::Japanese => "japanese",
            PpocrScriptClass::Kannada => "kannada",
            PpocrScriptClass::Korean => "korean",
            PpocrScriptClass::Tamil => "tamil",
            PpocrScriptClass::Telugu => "telugu",
            PpocrScriptClass::Latin => "latin",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PpocrScriptPrediction {
    pub class: PpocrScriptClass,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
struct PpocrScriptCandidate {
    class: PpocrScriptClass,
    score: f32,
}

const PUNCTUATIONS: &[char] = &[
    ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '_', '/', '\\',
    '|', '@', '#', '$', '%', '&', '*', '+', '=', '~', '，', '。', '！', '？', '；', '：', '、',
    '「', '」', '『', '』', '（', '）', '【', '】', '《', '》', '—', '…', '·', '～',
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PpocrRect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl PpocrRect {
    pub fn width(&self) -> u32 {
        self.right.saturating_sub(self.left)
    }
    pub fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }
}

pub struct PpocrDetector {
    session: MnnSession,
}

pub struct PpocrRecognizer {
    /// One MnnSession per parallel worker. Each is wrapped in a Mutex but contention is zero —
    /// the rayon worker at index `i` only touches `sessions[i]`. The Mutex is just there to
    /// satisfy Sync on MnnSession's interior mutability without making assumptions about MNN's
    /// thread-safety guarantees.
    sessions: Vec<Mutex<MnnSession>>,
    charset: Vec<char>,
    pool: ThreadPool,
}

pub struct PpocrScriptClassifier {
    session: MnnSession,
}

#[derive(Debug, Clone)]
pub struct PpocrRecognizerSpec {
    pub script: PpocrScript,
    pub model_path: PathBuf,
    pub keys_path: PathBuf,
}

struct PpocrRecognizerSlot {
    spec: PpocrRecognizerSpec,
    loaded: OnceLock<Arc<PpocrRecognizer>>,
}

pub struct PpocrEngine {
    detector: PpocrDetector,
    classifier: Option<PpocrScriptClassifier>,
    recognizers: HashMap<PpocrScript, PpocrRecognizerSlot>,
}

impl PpocrEngine {
    pub fn load(
        det_path: &Path,
        classifier_path: Option<&Path>,
        recognizer_specs: Vec<PpocrRecognizerSpec>,
        det_intra_threads: usize,
    ) -> Result<Self, TranslatorError> {
        // Det is one big graph and benefits from intra-session threading. Rec models are loaded
        // lazily per script so auto mode can route strips to multiple scripts without
        // constructing every recognizer up front.
        let detector = PpocrDetector::load(det_path, det_intra_threads)?;
        let classifier = classifier_path
            .map(PpocrScriptClassifier::load)
            .transpose()?;
        let recognizers = recognizer_specs
            .into_iter()
            .map(|spec| {
                (
                    spec.script,
                    PpocrRecognizerSlot {
                        spec,
                        loaded: OnceLock::new(),
                    },
                )
            })
            .collect();
        Ok(Self {
            detector,
            classifier,
            recognizers,
        })
    }

    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }

    pub fn installed_scripts(&self) -> impl Iterator<Item = PpocrScript> + '_ {
        self.recognizers.keys().copied()
    }

    fn recognizer(&self, script: PpocrScript) -> Result<Arc<PpocrRecognizer>, TranslatorError> {
        let slot = self.recognizers.get(&script).ok_or_else(|| {
            TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                format!(
                    "ppocr recognizer for script {} is not installed",
                    script.as_slug()
                ),
            )
        })?;
        if let Some(rec) = slot.loaded.get() {
            return Ok(Arc::clone(rec));
        }
        let fresh = Arc::new(PpocrRecognizer::load(
            &slot.spec.model_path,
            &slot.spec.keys_path,
        )?);
        Ok(Arc::clone(slot.loaded.get_or_init(|| fresh)))
    }

    /// Run the detector on a pre-built `DynamicImage` and return geometry only,
    /// without recognition. `profile` selects still vs live thresholds.
    pub fn detect_only_image(
        &self,
        image: &DynamicImage,
        profile: PpocrProfile,
    ) -> Result<Vec<crate::ocr::DetectedTextBox>, TranslatorError> {
        self.detect_only_image_with_thresholds(image, profile.thresholds())
    }

    fn detect_only_image_with_thresholds(
        &self,
        image: &DynamicImage,
        thresholds: PpocrThresholds,
    ) -> Result<Vec<crate::ocr::DetectedTextBox>, TranslatorError> {
        let width = image.width();
        let height = image.height();
        let boxes = self.detector.detect_with_thresholds(image, thresholds)?;
        let out: Vec<crate::ocr::DetectedTextBox> = boxes
            .into_iter()
            .map(|tb| {
                let expanded = expand_box(&tb.rect, DET_BOX_BORDER, width, height);
                let aabb = crate::ocr::Rect {
                    left: expanded.left,
                    top: expanded.top,
                    right: expanded.right,
                    bottom: expanded.bottom,
                };
                let contour_boxes = tb
                    .contour
                    .as_ref()
                    .and_then(|c| oriented_boxes_from_contour(c));
                let (oriented, tight) = match contour_boxes {
                    Some(ContourBoxes { tight, inflated }) => (inflated, tight),
                    None => {
                        let aligned = crate::ocr::OrientedRect::axis_aligned(aabb);
                        (aligned, aligned)
                    }
                };
                let contour_flat: Vec<f32> = tb
                    .contour
                    .as_ref()
                    .map(|c| {
                        let mut v = Vec::with_capacity(c.len() * 2);
                        for &(x, y) in c {
                            v.push(x);
                            v.push(y);
                        }
                        v
                    })
                    .unwrap_or_default();
                crate::ocr::DetectedTextBox {
                    rect: aabb,
                    oriented_box: oriented,
                    tight_box: tight,
                    contour: contour_flat,
                    score: tb.score,
                }
            })
            .collect();
        Ok(out)
    }

    pub fn classify_text_boxes_image(
        &self,
        image: &DynamicImage,
        gray: &GrayImage,
        boxes: &[crate::ocr::DetectedTextBox],
    ) -> Result<Vec<Option<PpocrScriptPrediction>>, TranslatorError> {
        let Some(classifier) = &self.classifier else {
            return Err(TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                "ppocr script classifier is not available",
            ));
        };
        let crops = crop_text_strips(image, gray, boxes).0;
        let image_area = (image.width() as f32) * (image.height() as f32);
        let mut predictions = vec![None; boxes.len()];
        let mut eligible_crops = Vec::new();
        let mut eligible_indices = Vec::new();
        for (idx, (box_, crop)) in boxes.iter().zip(crops.iter()).enumerate() {
            if pulc_strip_eligible(box_, crop, image_area) {
                eligible_indices.push(idx);
                eligible_crops.push(crop.clone());
            } else {
                let area = box_.rect.width().saturating_mul(box_.rect.height());
                log::debug!(
                    "ppocr pulc strip={} skipped small det_score={:.3} width={} height={} area={} area_ratio={:.6} min_area_ratio={:.6} crop={}x{}",
                    idx,
                    box_.score,
                    box_.rect.width(),
                    box_.rect.height(),
                    area,
                    area as f32 / image_area.max(1.0),
                    PULC_MIN_IMAGE_AREA_RATIO,
                    crop.width(),
                    crop.height(),
                );
            }
        }
        let classified = classifier.classify_many(&eligible_crops)?;
        for (idx, candidate) in eligible_indices.into_iter().zip(classified.into_iter()) {
            let box_ = &boxes[idx];
            let crop = &crops[idx];
            let area = box_.rect.width().saturating_mul(box_.rect.height());
            let area_ratio = area as f32 / image_area.max(1.0);
            let Some(candidate) = candidate else {
                log::debug!(
                    "ppocr pulc strip={} no_candidate det_score={:.3} width={} height={} area={} area_ratio={:.6} crop={}x{}",
                    idx,
                    box_.score,
                    box_.rect.width(),
                    box_.rect.height(),
                    area,
                    area_ratio,
                    crop.width(),
                    crop.height(),
                );
                continue;
            };
            if candidate.score >= PULC_MIN_SCORE {
                log::debug!(
                    "ppocr pulc strip={} script={} score={:.3} det_score={:.3} width={} height={} area={} area_ratio={:.6} crop={}x{}",
                    idx,
                    candidate.class.name(),
                    candidate.score,
                    box_.score,
                    box_.rect.width(),
                    box_.rect.height(),
                    area,
                    area_ratio,
                    crop.width(),
                    crop.height(),
                );
                predictions[idx] = Some(PpocrScriptPrediction {
                    class: candidate.class,
                    score: candidate.score,
                });
            } else {
                log::debug!(
                    "ppocr pulc strip={} script={} score={:.3} below_threshold={:.2} det_score={:.3} width={} height={} area={} area_ratio={:.6} crop={}x{}",
                    idx,
                    candidate.class.name(),
                    candidate.score,
                    PULC_MIN_SCORE,
                    box_.score,
                    box_.rect.width(),
                    box_.rect.height(),
                    area,
                    area_ratio,
                    crop.width(),
                    crop.height(),
                );
            }
        }
        Ok(predictions)
    }

    /// Recognize text in caller-supplied boxes. `scripts` selects the recognizer pack
    /// per box (parallel to `boxes`); strips with the same script are dispatched as a
    /// batch to a single recognizer session pool. `profile` selects still vs live
    /// thresholds. Output is 1:1 aligned with input boxes; filtered entries come back
    /// with empty text and `confidence = 0.0`.
    pub fn recognize_text_in_boxes_image(
        &self,
        image: &DynamicImage,
        gray: &GrayImage,
        boxes: &[crate::ocr::DetectedTextBox],
        scripts: &[PpocrScript],
        profile: PpocrProfile,
    ) -> Result<Vec<crate::ocr::RecognizedTextLine>, TranslatorError> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        if scripts.len() != boxes.len() {
            return Err(TranslatorError::new(
                TranslatorErrorKind::InvalidInput,
                "ppocr recognition scripts length must match boxes length",
            ));
        }
        let thresholds = profile.thresholds();
        let width = image.width();
        let height = image.height();

        let t_crops = Instant::now();
        let (crops, dewarp_count) = crop_text_strips(image, gray, boxes);
        let crops_ms = t_crops.elapsed().as_secs_f32() * 1000.0;
        let mean_crop_w: f32 =
            crops.iter().map(|c| c.width() as f32).sum::<f32>() / crops.len() as f32;
        let mean_crop_h: f32 =
            crops.iter().map(|c| c.height() as f32).sum::<f32>() / crops.len() as f32;

        let mut results: Vec<Option<RecResult>> = (0..boxes.len()).map(|_| None).collect();
        let mut rec_wall_ms = 0.0;
        let mut rec_pre_ms = 0.0;
        let mut rec_infer_ms = 0.0;
        let mut rec_post_ms = 0.0;
        let mut grouped = HashMap::<PpocrScript, Vec<usize>>::new();
        for (idx, script) in scripts.iter().enumerate() {
            grouped.entry(*script).or_default().push(idx);
        }
        for (script, indices) in grouped {
            let recognizer = self.recognizer(script)?;
            let timings_us = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
            let t_rec_wall = Instant::now();
            let group_results: Vec<Result<RecResult, TranslatorError>> =
                recognizer.pool.install(|| {
                    indices
                        .par_iter()
                        .map(|&idx| {
                            let worker_idx = rayon::current_thread_index().unwrap_or(0)
                                % recognizer.sessions.len();
                            recognizer.recognize_one(&crops[idx], worker_idx, &timings_us)
                        })
                        .collect()
                });
            rec_wall_ms += t_rec_wall.elapsed().as_secs_f32() * 1000.0;
            rec_pre_ms += timings_us.0.load(Ordering::Relaxed) as f32 / 1000.0;
            rec_infer_ms += timings_us.1.load(Ordering::Relaxed) as f32 / 1000.0;
            rec_post_ms += timings_us.2.load(Ordering::Relaxed) as f32 / 1000.0;
            for (idx, result) in indices.into_iter().zip(group_results.into_iter()) {
                results[idx] = Some(result?);
            }
        }

        let mut lines = Vec::with_capacity(boxes.len());
        let mut empty_count = 0usize;
        let mut low_score_count = 0usize;
        for (idx, result) in results.into_iter().enumerate() {
            let r = result.expect("all routed ppocr recognition results populated");
            let raw_text = r.text.trim().to_owned();
            let raw_confidence = r.confidence;
            let (text, confidence, status) = if raw_text.is_empty() {
                empty_count += 1;
                (String::new(), 0.0, "empty")
            } else if raw_confidence < thresholds.rec_drop_score {
                low_score_count += 1;
                (String::new(), 0.0, "low_score")
            } else {
                let accepted = match scripts[idx] {
                    PpocrScript::Cyrillic | PpocrScript::Eslav => {
                        crate::script_normalize::repair_cyrillic_word_mixing(&r.text)
                    }
                    _ => r.text,
                };
                (accepted, r.confidence, "accepted")
            };
            log::debug!(
                "ppocr rec strip={} script={} det_score={:.3} width={} height={} area={} conf={:.3} status={} text=\"{}\"",
                idx,
                scripts[idx].as_slug(),
                boxes[idx].score,
                boxes[idx].rect.width(),
                boxes[idx].rect.height(),
                boxes[idx]
                    .rect
                    .width()
                    .saturating_mul(boxes[idx].rect.height()),
                raw_confidence,
                status,
                log_text_preview(&raw_text),
            );
            lines.push(crate::ocr::RecognizedTextLine {
                rect: boxes[idx].rect,
                oriented_box: boxes[idx].oriented_box,
                text,
                confidence,
                source_code: None,
            });
        }
        log::debug!(
            "ppocr rec: src={}x{} boxes={} dewarped={}/{} mean_crop={:.0}x{:.0} \
             accepted={} empty={} low_score={} (drop {:.2}) — \
             crops/dewarp={:.1}ms \
             rec_wall={:.1}ms (cpu pre={:.1}ms infer={:.1}ms post={:.1}ms)",
            width,
            height,
            boxes.len(),
            dewarp_count,
            boxes.len(),
            mean_crop_w,
            mean_crop_h,
            lines.iter().filter(|l| !l.text.is_empty()).count(),
            empty_count,
            low_score_count,
            thresholds.rec_drop_score,
            crops_ms,
            rec_wall_ms,
            rec_pre_ms,
            rec_infer_ms,
            rec_post_ms,
        );
        Ok(lines)
    }
}

fn log_text_preview(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars().flat_map(char::escape_default).take(80) {
        out.push(ch);
    }
    out
}

fn pulc_strip_eligible(
    box_: &crate::ocr::DetectedTextBox,
    crop: &DynamicImage,
    image_area: f32,
) -> bool {
    let box_area = box_.rect.width().saturating_mul(box_.rect.height());
    let crop_area = crop.width().saturating_mul(crop.height());
    let box_area_ratio = box_area as f32 / image_area.max(1.0);
    box_.rect.width() >= PULC_MIN_STRIP_WIDTH
        && box_.rect.height() >= PULC_MIN_STRIP_HEIGHT
        && box_area >= PULC_MIN_STRIP_AREA
        && box_area_ratio >= PULC_MIN_IMAGE_AREA_RATIO
        && crop.width() >= PULC_MIN_STRIP_WIDTH
        && crop.height() >= PULC_MIN_STRIP_HEIGHT
        && crop_area >= PULC_MIN_STRIP_AREA
}

pub(crate) fn rgba_to_dynamic(rgba: &[u8], width: u32, height: u32) -> DynamicImage {
    let n_pixels = (width as usize) * (height as usize);
    let mut rgb = Vec::with_capacity(n_pixels * 3);
    for i in 0..n_pixels {
        let base = i * 4;
        rgb.push(rgba[base]);
        rgb.push(rgba[base + 1]);
        rgb.push(rgba[base + 2]);
    }
    let img = RgbImage::from_raw(width, height, rgb).expect("rgb buffer sized correctly");
    DynamicImage::ImageRgb8(img)
}

fn crop_dynamic(image: &DynamicImage, rect: &PpocrRect) -> DynamicImage {
    let w = rect.width().max(1);
    let h = rect.height().max(1);
    image.crop_imm(rect.left, rect.top, w, h)
}

fn crop_text_strips(
    image: &DynamicImage,
    gray: &GrayImage,
    boxes: &[crate::ocr::DetectedTextBox],
) -> (Vec<DynamicImage>, usize) {
    let mut dewarp_count = 0usize;
    let crops = boxes
        .iter()
        .map(|b| {
            let contour_pairs: Option<Vec<(f32, f32)>> =
                if !b.contour.is_empty() && b.contour.len() % 2 == 0 {
                    Some(b.contour.chunks_exact(2).map(|c| (c[0], c[1])).collect())
                } else {
                    None
                };
            let dewarped = contour_pairs
                .as_deref()
                .and_then(|c| dewarp_contour_to_strip(gray, c))
                .map(DynamicImage::ImageLuma8);
            if let Some(dewarped) = dewarped {
                dewarp_count += 1;
                dewarped
            } else {
                let rect = PpocrRect {
                    left: b.rect.left,
                    top: b.rect.top,
                    right: b.rect.right,
                    bottom: b.rect.bottom,
                };
                crop_dynamic(image, &rect)
            }
        })
        .collect();
    (crops, dewarp_count)
}

fn expand_box(rect: &PpocrRect, border: u32, max_w: u32, max_h: u32) -> PpocrRect {
    PpocrRect {
        left: rect.left.saturating_sub(border),
        top: rect.top.saturating_sub(border),
        right: (rect.right + border).min(max_w),
        bottom: (rect.bottom + border).min(max_h),
    }
}

// ---------- Detector ----------

#[derive(Debug, Clone)]
struct DetBox {
    rect: PpocrRect,
    contour: Option<Vec<(f32, f32)>>,
    score: f32,
}

impl PpocrDetector {
    fn load(model_path: &Path, intra_threads: usize) -> Result<Self, TranslatorError> {
        let session = MnnSession::load(model_path, intra_threads)?;
        Ok(Self { session })
    }

    fn detect_with_thresholds(
        &self,
        image: &DynamicImage,
        thresholds: PpocrThresholds,
    ) -> Result<Vec<DetBox>, TranslatorError> {
        let (orig_w, orig_h) = image.dimensions();
        let t_pre = Instant::now();
        let scaled = resize_to_max_side(image, DET_MAX_SIDE);
        let (scaled_w, scaled_h) = scaled.dimensions();
        let pad_w = pad_to_multiple(scaled_w, 32);
        let pad_h = pad_to_multiple(scaled_h, 32);
        let tensor_buf = preprocess_for_det(&scaled, pad_w, pad_h);
        let pre_ms = t_pre.elapsed().as_secs_f32() * 1000.0;

        let t_infer = Instant::now();
        let (mask, out_shape) = self
            .session
            .run(&tensor_buf, &[1, 3, pad_h as usize, pad_w as usize])?;
        let infer_ms = t_infer.elapsed().as_secs_f32() * 1000.0;

        if out_shape.len() < 4 {
            return Err(TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!("ppocr det output shape unexpected: {:?}", out_shape),
            ));
        }
        let out_h = out_shape[out_shape.len() - 2] as u32;
        let out_w = out_shape[out_shape.len() - 1] as u32;
        let mut mask_min = f32::INFINITY;
        let mut mask_max = f32::NEG_INFINITY;
        let mut mask_sum = 0.0f32;
        let mut over_thresh = 0usize;
        for &v in &mask {
            if v < mask_min {
                mask_min = v;
            }
            if v > mask_max {
                mask_max = v;
            }
            mask_sum += v;
            if v > thresholds.det_score_threshold {
                over_thresh += 1;
            }
        }
        let mask_mean = mask_sum / mask.len() as f32;
        let t_post = Instant::now();
        let binary: Vec<u8> = mask
            .iter()
            .map(|&v| {
                if v > thresholds.det_score_threshold {
                    255
                } else {
                    0
                }
            })
            .collect();
        let boxes = extract_boxes(
            &binary, &mask, out_w, out_h, scaled_w, scaled_h, orig_w, orig_h, thresholds,
        );
        let post_ms = t_post.elapsed().as_secs_f32() * 1000.0;
        log::debug!(
            "ppocr det: input_pad={}x{} scaled={}x{} out={}x{} mask[min/max/mean]={:.3}/{:.3}/{:.3} over_{}={}/{} — pre={:.1}ms infer={:.1}ms post={:.1}ms",
            pad_w,
            pad_h,
            scaled_w,
            scaled_h,
            out_w,
            out_h,
            mask_min,
            mask_max,
            mask_mean,
            thresholds.det_score_threshold,
            over_thresh,
            mask.len(),
            pre_ms,
            infer_ms,
            post_ms,
        );
        Ok(boxes)
    }
}

impl PpocrScriptClassifier {
    fn load(model_path: &Path) -> Result<Self, TranslatorError> {
        let session =
            MnnSession::load_with_modes(model_path, 4, PrecisionMode::Low, MemoryMode::Low)?;
        Ok(Self { session })
    }

    fn classify_many(
        &self,
        crops: &[DynamicImage],
    ) -> Result<Vec<Option<PpocrScriptCandidate>>, TranslatorError> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let input = preprocess_for_pulc(crops);
        let (out, out_shape) = self.session.run(
            &input,
            &[crops.len(), 3, PULC_HEIGHT as usize, PULC_WIDTH as usize],
        )?;
        if out_shape.len() != 2 || out_shape[0] != crops.len() || out_shape[1] != PULC_CLASSES.len()
        {
            return Err(TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!(
                    "ppocr script classifier output shape unexpected: {:?}",
                    out_shape
                ),
            ));
        }
        Ok(out
            .chunks_exact(PULC_CLASSES.len())
            .map(top_pulc_candidate)
            .collect())
    }
}

fn pad_to_multiple(v: u32, m: u32) -> u32 {
    v.div_ceil(m) * m
}

fn resize_to_max_side(image: &DynamicImage, max_side: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    let max_dim = w.max(h);
    if max_dim <= max_side {
        return image.clone();
    }
    let scale = max_side as f32 / max_dim as f32;
    let nw = (w as f32 * scale).round().max(1.0) as u32;
    let nh = (h as f32 * scale).round().max(1.0) as u32;
    image.resize_exact(nw, nh, FilterType::Triangle)
}

fn preprocess_for_det(image: &DynamicImage, pad_w: u32, pad_h: u32) -> Vec<f32> {
    let plane = (pad_w as usize) * (pad_h as usize);
    let mut buf = vec![0.0f32; 3 * plane];
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let pad_w_u = pad_w as usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let idx = y * pad_w_u + x;
            buf[idx] = (pixel[0] as f32 / 255.0 - PPOCR_DET_MEAN[0]) / PPOCR_DET_STD[0];
            buf[plane + idx] = (pixel[1] as f32 / 255.0 - PPOCR_DET_MEAN[1]) / PPOCR_DET_STD[1];
            buf[2 * plane + idx] = (pixel[2] as f32 / 255.0 - PPOCR_DET_MEAN[2]) / PPOCR_DET_STD[2];
        }
    }
    buf
}

fn preprocess_for_pulc(crops: &[DynamicImage]) -> Vec<f32> {
    let plane = (PULC_HEIGHT as usize) * (PULC_WIDTH as usize);
    let mut buf = vec![0.0f32; crops.len() * 3 * plane];
    for (batch, crop) in crops.iter().enumerate() {
        let resized = crop.resize_exact(PULC_WIDTH, PULC_HEIGHT, FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let batch_base = batch * 3 * plane;
        for y in 0..PULC_HEIGHT as usize {
            for x in 0..PULC_WIDTH as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = y * PULC_WIDTH as usize + x;
                buf[batch_base + idx] =
                    (pixel[0] as f32 / 255.0 - PPOCR_DET_MEAN[0]) / PPOCR_DET_STD[0];
                buf[batch_base + plane + idx] =
                    (pixel[1] as f32 / 255.0 - PPOCR_DET_MEAN[1]) / PPOCR_DET_STD[1];
                buf[batch_base + 2 * plane + idx] =
                    (pixel[2] as f32 / 255.0 - PPOCR_DET_MEAN[2]) / PPOCR_DET_STD[2];
            }
        }
    }
    buf
}

fn top_pulc_candidate(row: &[f32]) -> Option<PpocrScriptCandidate> {
    let probs = if row.iter().all(|v| v.is_finite()) {
        let sum = row.iter().sum::<f32>();
        if (0.95..=1.05).contains(&sum) {
            row.to_vec()
        } else {
            softmax(row)
        }
    } else {
        return None;
    };
    let (idx, score) = probs
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;
    Some(PpocrScriptCandidate {
        class: PULC_CLASSES[idx],
        score,
    })
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|v| v / sum).collect()
}

fn extract_boxes(
    mask: &[u8],
    heatmap: &[f32],
    mask_w: u32,
    mask_h: u32,
    valid_w: u32,
    valid_h: u32,
    orig_w: u32,
    orig_h: u32,
    thresholds: PpocrThresholds,
) -> Vec<DetBox> {
    let Some(gray) = GrayImage::from_raw(mask_w, mask_h, mask.to_vec()) else {
        return Vec::new();
    };
    let contours = find_contours::<i32>(&gray);
    let scale_x = orig_w as f32 / valid_w as f32;
    let scale_y = orig_h as f32 / valid_h as f32;
    let mut boxes = Vec::new();
    let mut weak_score_count = 0usize;

    for contour in contours {
        if contour.parent.is_some() || contour.points.len() < 4 {
            continue;
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for p in &contour.points {
            if p.x < min_x {
                min_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            }
            if p.x > max_x {
                max_x = p.x;
            }
            if p.y > max_y {
                max_y = p.y;
            }
        }
        if min_x >= valid_w as i32 || min_y >= valid_h as i32 {
            continue;
        }
        let min_x = min_x.max(0);
        let min_y = min_y.max(0);
        let max_x = max_x.min(valid_w as i32);
        let max_y = max_y.min(valid_h as i32);
        let box_w = (max_x - min_x) as u32;
        let box_h = (max_y - min_y) as u32;
        if box_w * box_h < thresholds.det_min_area {
            continue;
        }

        // Box-score gate (PaddleOCR's `det_db_box_thresh`): mean heatmap probability over the
        // contour's *binarised interior* on the raw float output. We restrict the average to
        // pixels above `DET_SCORE_THRESHOLD` (i.e. pixels that the binarised mask considers
        // text) rather than the full AABB, because the AABB of a tilted line contains a lot
        // of background corner pixels that would dilute the mean — penalising tilted text
        // unfairly. This is effectively the polygon-interior mode upstream calls "slow",
        // approximated cheaply with the binary mask we already have. Real text masks average
        // 0.7+ here; texture/compression noise that just barely cleared the per-pixel
        // threshold rarely makes it past 0.5.
        let mut score_sum = 0.0f32;
        let mut score_n = 0usize;
        for y in (min_y as u32)..(max_y as u32).min(mask_h) {
            let row = (y as usize) * (mask_w as usize);
            for x in (min_x as u32)..(max_x as u32).min(mask_w) {
                let idx = row + x as usize;
                if mask[idx] != 0 {
                    score_sum += heatmap[idx];
                    score_n += 1;
                }
            }
        }
        let box_score = if score_n > 0 {
            score_sum / score_n as f32
        } else {
            0.0
        };
        if box_score < thresholds.det_box_min_score {
            weak_score_count += 1;
            continue;
        }

        // DB unclip: distance = area * ratio / perimeter.
        let area = box_w as f32 * box_h as f32;
        let perimeter = 2.0 * (box_w + box_h) as f32;
        let expand_dist = (area * DET_UNCLIP_RATIO / perimeter).max(1.0);
        let ex_min_x = (min_x as f32 - expand_dist).max(0.0) as i32;
        let ex_min_y = (min_y as f32 - expand_dist).max(0.0) as i32;
        let ex_max_x = (max_x as f32 + expand_dist).min(valid_w as f32) as i32;
        let ex_max_y = (max_y as f32 + expand_dist).min(valid_h as f32) as i32;
        let ex_w = (ex_max_x - ex_min_x) as u32;
        let ex_h = (ex_max_y - ex_min_y) as u32;

        let scaled_x = (ex_min_x as f32 * scale_x) as i32;
        let scaled_y = (ex_min_y as f32 * scale_y) as i32;
        let scaled_w = (ex_w as f32 * scale_x) as u32;
        let scaled_h = (ex_h as f32 * scale_y) as u32;
        let final_x = scaled_x.max(0) as u32;
        let final_y = scaled_y.max(0) as u32;
        let final_w = scaled_w.min(orig_w.saturating_sub(final_x));
        let final_h = scaled_h.min(orig_h.saturating_sub(final_y));
        if final_w == 0 || final_h == 0 {
            continue;
        }

        let scaled_contour: Vec<(f32, f32)> = contour
            .points
            .iter()
            .map(|p: &Point<i32>| (p.x as f32 * scale_x, p.y as f32 * scale_y))
            .collect();

        boxes.push(DetBox {
            rect: PpocrRect {
                left: final_x,
                top: final_y,
                right: final_x + final_w,
                bottom: final_y + final_h,
            },
            contour: Some(scaled_contour),
            score: box_score,
        });
    }
    if weak_score_count > 0 {
        log::info!(
            "ppocr det: {} contour(s) below box-score gate {:.2}",
            weak_score_count,
            thresholds.det_box_min_score,
        );
    }
    boxes
}

// ---------- Recognizer ----------

struct RecResult {
    text: String,
    confidence: f32,
}

impl PpocrRecognizer {
    fn load(model_path: &Path, keys_path: &Path) -> Result<Self, TranslatorError> {
        let mut sessions = Vec::with_capacity(REC_PARALLELISM);
        for _ in 0..REC_PARALLELISM {
            sessions.push(Mutex::new(MnnSession::load(model_path, 1)?));
        }
        let charset = load_charset(keys_path)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(REC_PARALLELISM)
            .thread_name(|i| format!("ppocr-rec-{i}"))
            .build()
            .map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("failed to build ppocr rec thread pool: {e}"),
                )
            })?;
        Ok(Self {
            sessions,
            charset,
            pool,
        })
    }

    /// Run rec on a single crop using `sessions[session_idx]`. Each call is a batch-of-1 MNN
    /// inference at the crop's actual aspect ratio — no padding waste.
    fn recognize_one(
        &self,
        image: &DynamicImage,
        session_idx: usize,
        timings_us: &(AtomicU64, AtomicU64, AtomicU64),
    ) -> Result<RecResult, TranslatorError> {
        let t_pre = Instant::now();
        let target_h = REC_TARGET_HEIGHT as usize;
        let (orig_w, orig_h) = image.dimensions();
        let scale = REC_TARGET_HEIGHT as f32 / orig_h as f32;
        let sw = ((orig_w as f32 * scale).round().max(1.0)) as u32;

        let resized = image.resize_exact(sw, REC_TARGET_HEIGHT, FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let w_us = sw as usize;
        let mut buf = vec![0.0f32; 3 * target_h * w_us];
        let plane = target_h * w_us;
        for y in 0..target_h {
            for x in 0..w_us {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = y * w_us + x;
                buf[idx] = (pixel[0] as f32 / 255.0 - PPOCR_REC_MEAN[0]) / PPOCR_REC_STD[0];
                buf[plane + idx] = (pixel[1] as f32 / 255.0 - PPOCR_REC_MEAN[1]) / PPOCR_REC_STD[1];
                buf[2 * plane + idx] =
                    (pixel[2] as f32 / 255.0 - PPOCR_REC_MEAN[2]) / PPOCR_REC_STD[2];
            }
        }
        timings_us
            .0
            .fetch_add(t_pre.elapsed().as_micros() as u64, Ordering::Relaxed);

        let t_infer = Instant::now();
        let (out_data, out_shape) = {
            let session = self.sessions[session_idx]
                .lock()
                .expect("ppocr rec session mutex poisoned");
            session.run(&buf, &[1, 3, target_h, w_us])?
        };
        timings_us
            .1
            .fetch_add(t_infer.elapsed().as_micros() as u64, Ordering::Relaxed);

        if out_shape.len() != 3 || out_shape[0] != 1 {
            return Err(TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!("ppocr rec output shape unexpected: {:?}", out_shape),
            ));
        }
        let t_post = Instant::now();
        let seq_len = out_shape[1];
        let num_classes = out_shape[2];
        let result = decode_ctc(&out_data, seq_len, num_classes, &self.charset);
        timings_us
            .2
            .fetch_add(t_post.elapsed().as_micros() as u64, Ordering::Relaxed);
        Ok(result)
    }
}

fn load_charset(path: &Path) -> Result<Vec<char>, TranslatorError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        TranslatorError::new(
            TranslatorErrorKind::Internal,
            format!("failed to read ppocr charset {}: {e}", path.display()),
        )
    })?;
    // PaddleOCR conventions: index 0 is blank, last index is padding/space, so we wrap with
    // a leading blank and a trailing pad character (' ').
    let mut charset = Vec::with_capacity(content.chars().count() + 2);
    charset.push(' ');
    for line in content.lines() {
        if let Some(ch) = line.chars().next() {
            charset.push(ch);
        }
    }
    charset.push(' ');
    if charset.len() < 3 {
        return Err(TranslatorError::new(
            TranslatorErrorKind::Internal,
            format!("ppocr charset {} is too small", path.display()),
        ));
    }
    Ok(charset)
}

fn decode_ctc(logits: &[f32], seq_len: usize, num_classes: usize, charset: &[char]) -> RecResult {
    let mut chars: Vec<(char, f32)> = Vec::new();
    let mut prev_idx = 0usize;
    for t in 0..seq_len {
        let start = t * num_classes;
        let end = start + num_classes;
        let probs = &logits[start..end];
        let mut max_idx = 0usize;
        let mut max_prob = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                max_idx = i;
            }
        }
        if max_idx != 0 && max_idx != prev_idx && max_idx < charset.len() {
            let ch = charset[max_idx];
            let threshold = if PUNCTUATIONS.contains(&ch) {
                REC_PUNCT_MIN_SCORE
            } else {
                REC_MIN_SCORE
            };
            if max_prob >= threshold {
                chars.push((ch, max_prob));
            }
        }
        prev_idx = max_idx;
    }
    let confidence = if chars.is_empty() {
        0.0
    } else {
        chars.iter().map(|(_, s)| *s).sum::<f32>() / chars.len() as f32
    };
    let text: String = chars.iter().map(|(c, _)| *c).collect();
    RecResult { text, confidence }
}

// ---------- Per-line contour-based dewarp (ported from OCR PoC) ----------

/// Compute the min-area-aligned (PCA-axis) rotated rectangle around a detection contour. The
/// principal axis is the line's reading direction; perpendicular extent gives the height. For
/// elongated text shapes this is within a fraction of a degree of the true min-area rect, but
/// is much cheaper than the rotating-calipers algorithm.
#[derive(Debug, Clone, Copy)]
struct ContourBoxes {
    /// Min-area rotated rectangle around the *raw* DB mask contour. Tight to the segmentation
    /// kernel — no ascender/descender padding, no unclip inflation. Used for layout heuristics
    /// (paragraph grouping, line-height clustering) where character whitespace would skew the
    /// metric.
    tight: crate::ocr::OrientedRect,
    /// `tight` inflated by `unclip + DET_BOX_BORDER`, matching the AABB pipeline. Used for
    /// erase/render so the box covers actual ink including ascenders/descenders.
    inflated: crate::ocr::OrientedRect,
}

/// Number of x-bins used to extract the top and bottom edge profile from a contour. 16 is
/// enough to fit a regression through a real word's edges while staying coarse enough that a
/// single noisy contour point doesn't dominate any one bin's value.
const TILT_X_BINS: usize = 16;

/// Maximum |top_slope − bottom_slope| (in dy/dx units) for the contour to be considered
/// genuinely tilted. ~0.05 ≈ 2.9° — well below typical glyph-asymmetry slope ("Menu"'s top
/// edge drops 30% of its height across the word, slope ≈ 0.10–0.20) but above any real-world
/// scanning skew jitter.
const TILT_AGREEMENT_SLOPE: f32 = 0.05;

/// Per-bin extreme-y tolerance for outlier filtering, expressed as a fraction of the contour's
/// vertical extent. Descenders / ascenders / random spikes stick out beyond the median edge by
/// more than this and get dropped from the regression.
const TILT_EDGE_OUTLIER_FRACTION: f32 = 0.20;

/// Minimum end-to-end y-deviation (slope × contour width), expressed as a fraction of the
/// contour's height, for a tilt to be reported as non-zero. Short asymmetric contours like a
/// 4-letter word with a cap-height opener bias both the top and bottom regressions slightly
/// downward — the agreement gate then averages two biased numbers into a phantom tilt. By
/// also requiring the actual y-deviation across the contour to be ≥ 20% of its height, we
/// reject sub-visible tilts where the per-edge slopes likely come from regression artefacts
/// rather than from a genuine baseline lean.
const TILT_MIN_DEVIATION_FRACTION: f32 = 0.20;

/// Estimate a contour's horizontal tilt by regressing the top and bottom edges separately and
/// only trusting a non-zero angle when both edges agree. Returns the angle in radians (+x ⇒ 0,
/// downward-to-the-right ⇒ positive, image y points down).
///
/// Algorithm:
///   1. Bin contour points by x.
///   2. Per bin, take the minimum-y point (top edge) and maximum-y point (bottom edge).
///   3. Discard bins whose extreme-y deviates from the median extreme-y by more than
///      `TILT_EDGE_OUTLIER_FRACTION × contour_height` (handles ascenders / descenders /
///      asymmetric capital opener like "Menu"'s `M`).
///   4. Linear-regress slope on the filtered top points and on the filtered bottom points.
///   5. If the two slopes differ by more than `TILT_AGREEMENT_SLOPE`, the contour is
///      content-asymmetric and we report no tilt. Otherwise the average is the line's tilt.
fn estimate_horizontal_tilt(contour: &[(f32, f32)]) -> f32 {
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(x, y) in contour {
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let width = max_x - min_x;
    let height = max_y - min_y;
    if width < 8.0 || height < 2.0 {
        return 0.0;
    }

    let bin_w = width / TILT_X_BINS as f32;
    let mut top_per_bin: [Option<(f32, f32)>; TILT_X_BINS] = [None; TILT_X_BINS];
    let mut bot_per_bin: [Option<(f32, f32)>; TILT_X_BINS] = [None; TILT_X_BINS];
    for &(x, y) in contour {
        let bin = (((x - min_x) / bin_w) as usize).min(TILT_X_BINS - 1);
        match top_per_bin[bin] {
            None => top_per_bin[bin] = Some((x, y)),
            Some((_, ty)) if y < ty => top_per_bin[bin] = Some((x, y)),
            _ => {}
        }
        match bot_per_bin[bin] {
            None => bot_per_bin[bin] = Some((x, y)),
            Some((_, by)) if y > by => bot_per_bin[bin] = Some((x, y)),
            _ => {}
        }
    }

    let top_pts: Vec<(f32, f32)> = top_per_bin.iter().filter_map(|p| *p).collect();
    let bot_pts: Vec<(f32, f32)> = bot_per_bin.iter().filter_map(|p| *p).collect();
    if top_pts.len() < 4 || bot_pts.len() < 4 {
        return 0.0;
    }

    let tol = height * TILT_EDGE_OUTLIER_FRACTION;
    let top_slope = robust_regression_slope(&top_pts, tol);
    let bot_slope = robust_regression_slope(&bot_pts, tol);
    let diff = (top_slope - bot_slope).abs();
    let (angle, decision) = if diff > TILT_AGREEMENT_SLOPE {
        (0.0, "disagree → axis-align")
    } else {
        let avg = (top_slope + bot_slope) * 0.5;
        // Sub-visible tilt rejection: even when top and bot slopes "agree", if the resulting
        // end-to-end y-deviation is small relative to the contour height, the agreement is
        // probably just two same-direction regression biases (asymmetric cap on top + AA
        // jitter on bottom) lining up — not a real lean. Demand a visible deviation.
        if avg.abs() * width < TILT_MIN_DEVIATION_FRACTION * height {
            (0.0, "sub-visible → axis-align")
        } else {
            (avg.atan(), "agree → use avg")
        }
    };
    log::debug!(
        "ppocr tilt: w={:.0} h={:.0} top_bins={} bot_bins={} \
         top_slope={:.4} bot_slope={:.4} diff={:.4} (limit {:.4}) → {:.2}° ({})",
        width,
        height,
        top_pts.len(),
        bot_pts.len(),
        top_slope,
        bot_slope,
        diff,
        TILT_AGREEMENT_SLOPE,
        angle.to_degrees(),
        decision,
    );
    angle
}

/// Two-pass OLS with a residual-based outlier filter. The first pass captures the dominant
/// linear trend through the points; the second pass refits after dropping points whose
/// residual from the first-pass line exceeds `tol`. This is what makes the estimator robust
/// to descenders/ascenders that protrude from an otherwise straight edge while *keeping* the
/// edge's tilted endpoints (a median-based filter would discard them as "far from the centre"
/// which is exactly the position they should occupy in a tilted line).
fn robust_regression_slope(pts: &[(f32, f32)], tol: f32) -> f32 {
    if pts.len() < 3 {
        return 0.0;
    }
    let first = linear_regression_slope(pts);
    let mean_x = pts.iter().map(|(x, _)| x).sum::<f32>() / pts.len() as f32;
    let mean_y = pts.iter().map(|(_, y)| y).sum::<f32>() / pts.len() as f32;
    let intercept = mean_y - first * mean_x;
    let filtered: Vec<(f32, f32)> = pts
        .iter()
        .copied()
        .filter(|&(x, y)| (y - (first * x + intercept)).abs() <= tol)
        .collect();
    if filtered.len() < 3 {
        return first;
    }
    linear_regression_slope(&filtered)
}

fn linear_regression_slope(pts: &[(f32, f32)]) -> f32 {
    if pts.len() < 2 {
        return 0.0;
    }
    let n = pts.len() as f32;
    let mean_x = pts.iter().map(|(x, _)| x).sum::<f32>() / n;
    let mean_y = pts.iter().map(|(_, y)| y).sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for &(x, y) in pts {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den < 1e-6 { 0.0 } else { num / den }
}

fn oriented_boxes_from_contour(contour: &[(f32, f32)]) -> Option<ContourBoxes> {
    if contour.len() < 4 {
        return None;
    }
    let n = contour.len() as f32;

    let mut mean_x = 0.0f32;
    let mut mean_y = 0.0f32;
    for &(x, y) in contour {
        mean_x += x;
        mean_y += y;
    }
    mean_x /= n;
    mean_y /= n;

    // Estimate the line's tilt from the top *and* bottom edges of the contour separately, and
    // accept a non-zero angle only when both edges agree. PCA on all points (the previous
    // approach) is fooled by content-asymmetric masks — e.g. "Menu" has a tall M and short
    // enu, so the top edge slopes down while the baseline stays flat, and PCA's covariance
    // splits the difference into a phantom tilt. The bottom edge is the baseline (stable for
    // most words), and demanding agreement with the top filters out asymmetric shapes.
    let angle_radians = estimate_horizontal_tilt(contour);
    let ux = angle_radians.cos();
    let uy = angle_radians.sin();
    let vx = -uy;
    let vy = ux;

    let mut u_min = f32::INFINITY;
    let mut u_max = f32::NEG_INFINITY;
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    for &(x, y) in contour {
        let dx = x - mean_x;
        let dy = y - mean_y;
        let u = dx * ux + dy * uy;
        let v = dx * vx + dy * vy;
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    let raw_width = u_max - u_min;
    let raw_height = v_max - v_min;
    if raw_width < 4.0 || raw_height < 2.0 {
        return None;
    }
    let u_center = (u_min + u_max) * 0.5;
    let v_center = (v_min + v_max) * 0.5;
    let cx = mean_x + ux * u_center + vx * v_center;
    let cy = mean_y + uy * u_center + vy * v_center;

    let tight = crate::ocr::OrientedRect {
        cx,
        cy,
        width: raw_width,
        height: raw_height,
        angle_radians,
    };

    // DB segmentation produces a *shrunken* contour relative to the actual ink. The AABB path
    // recovers the full text region by expanding by `expand_dist = area * UNCLIP / perimeter`
    // plus a small border (see `extract_boxes` and `expand_box`). Apply the same inflation
    // here so the oriented rect covers ascenders/descenders for erase, and so its height
    // matches the AABB pipeline's height — what the renderer uses to size the font.
    let area = raw_width * raw_height;
    let perimeter = 2.0 * (raw_width + raw_height);
    let expand_dist = (area * DET_UNCLIP_RATIO / perimeter).max(1.0);
    let pad = expand_dist + DET_BOX_BORDER as f32;
    let inflated = crate::ocr::OrientedRect {
        cx,
        cy,
        width: raw_width + 2.0 * pad,
        height: raw_height + 2.0 * pad,
        angle_radians,
    };

    Some(ContourBoxes { tight, inflated })
}

fn dewarp_contour_to_strip(gray: &GrayImage, contour: &[(f32, f32)]) -> Option<GrayImage> {
    if contour.len() < 8 {
        return None;
    }
    let n = contour.len() as f32;

    // 1. PCA on contour points -> principal axis (u) and perpendicular (v).
    let mut mean_x = 0.0f32;
    let mut mean_y = 0.0f32;
    for &(x, y) in contour {
        mean_x += x;
        mean_y += y;
    }
    mean_x /= n;
    mean_y /= n;

    let mut cxx = 0.0f32;
    let mut cyy = 0.0f32;
    let mut cxy = 0.0f32;
    for &(x, y) in contour {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= n;
    cyy /= n;
    cxy /= n;

    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda1 = (trace + disc) * 0.5;
    let (ex, ey) = if cxy.abs() > 1e-6 {
        (lambda1 - cyy, cxy)
    } else if cxx >= cyy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let norm = (ex * ex + ey * ey).sqrt().max(1e-6);
    let mut ux = ex / norm;
    let mut uy = ey / norm;
    if ux < 0.0 {
        ux = -ux;
        uy = -uy;
    }
    let vx = -uy;
    let vy = ux;

    // 2. Project contour into (u, v) frame.
    let projected: Vec<(f32, f32)> = contour
        .iter()
        .map(|&(x, y)| {
            let dx = x - mean_x;
            let dy = y - mean_y;
            (dx * ux + dy * uy, dx * vx + dy * vy)
        })
        .collect();
    let (u_min, u_max) = projected
        .iter()
        .map(|&(u, _)| u)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), u| {
            (lo.min(u), hi.max(u))
        });
    let u_span = u_max - u_min;
    if u_span < 8.0 {
        return None;
    }

    // 3. Split contour by top/bottom half, fit quadratics. u normalized to [-1, 1] for
    //    numerical stability before fitting.
    let u_half_span = u_span * 0.5;
    let u_center = (u_min + u_max) * 0.5;
    let normalized: Vec<(f32, f32, f32)> = projected
        .iter()
        .map(|&(u, v)| ((u - u_center) / u_half_span, u, v))
        .collect();
    let top_pts: Vec<(f32, f32)> = normalized
        .iter()
        .filter(|&&(_, _, v)| v <= 0.0)
        .map(|&(un, _, v)| (un, v))
        .collect();
    let bot_pts: Vec<(f32, f32)> = normalized
        .iter()
        .filter(|&&(_, _, v)| v >= 0.0)
        .map(|&(un, _, v)| (un, v))
        .collect();
    let top_fit = fit_quadratic(&top_pts)?;
    let bot_fit = fit_quadratic(&bot_pts)?;

    // 4. Mean thickness for strip height; 2.4x inflation to give margin above ascenders /
    //    below descenders (PP-OCR was trained with whitespace padding around text).
    let n_samples = 32usize;
    let mut total_thickness = 0.0f32;
    for i in 0..n_samples {
        let un = -1.0 + 2.0 * (i as f32 / (n_samples - 1) as f32);
        total_thickness += eval_quadratic(bot_fit, un) - eval_quadratic(top_fit, un);
    }
    let mean_thickness = (total_thickness / n_samples as f32).max(1.0);
    let global_thickness = mean_thickness * 2.4;
    let u_pad = mean_thickness;
    let padded_u_min = u_min - u_pad;
    let padded_u_span = u_span + 2.0 * u_pad;
    let strip_h = (global_thickness.round() as u32).clamp(8, 256);
    let strip_w = (padded_u_span.round() as u32).clamp(16, 4096);

    // 5. Inverse warp: sample image along the curving spine, output to a rectangular strip.
    let mut out_buf = vec![0u8; (strip_w * strip_h) as usize];
    let gray_w = gray.width() as f32;
    let gray_h = gray.height() as f32;
    let gray_raw = gray.as_raw();
    let stride = gray.width() as usize;
    for x in 0..strip_w {
        let u_local = padded_u_min + (x as f32 + 0.5) * (padded_u_span / strip_w as f32);
        let un = (u_local - u_center) / u_half_span;
        let t = eval_quadratic(top_fit, un);
        let b = eval_quadratic(bot_fit, un);
        let spine_v = (t + b) * 0.5;
        for y in 0..strip_h {
            let v_norm = (y as f32 + 0.5) / strip_h as f32 - 0.5;
            let v_local = spine_v + v_norm * global_thickness;
            let src_x = mean_x + u_local * ux + v_local * vx;
            let src_y = mean_y + u_local * uy + v_local * vy;
            let sample = bilinear_luma(gray_raw, stride, gray_w, gray_h, src_x, src_y);
            out_buf[(y * strip_w + x) as usize] = sample;
        }
    }
    GrayImage::from_raw(strip_w, strip_h, out_buf)
}

fn fit_quadratic(points: &[(f32, f32)]) -> Option<(f32, f32, f32)> {
    if points.len() < 3 {
        return None;
    }
    let mut s = [0.0f64; 5];
    let mut t = [0.0f64; 3];
    s[0] = points.len() as f64;
    for &(u, v) in points {
        let u = u as f64;
        let v = v as f64;
        let u2 = u * u;
        s[1] += u;
        s[2] += u2;
        s[3] += u2 * u;
        s[4] += u2 * u2;
        t[0] += v;
        t[1] += u * v;
        t[2] += u2 * v;
    }
    let m = [[s[4], s[3], s[2]], [s[3], s[2], s[1]], [s[2], s[1], s[0]]];
    let inv = invert_3x3(&m)?;
    let a = inv[0][0] * t[2] + inv[0][1] * t[1] + inv[0][2] * t[0];
    let b = inv[1][0] * t[2] + inv[1][1] * t[1] + inv[1][2] * t[0];
    let c = inv[2][0] * t[2] + inv[2][1] * t[1] + inv[2][2] * t[0];
    Some((a as f32, b as f32, c as f32))
}

fn eval_quadratic(coeffs: (f32, f32, f32), u: f32) -> f32 {
    let (a, b, c) = coeffs;
    a * u * u + b * u + c
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

fn bilinear_luma(raw: &[u8], stride: usize, w: f32, h: f32, x: f32, y: f32) -> u8 {
    if x < 0.0 || y < 0.0 || x > w - 1.0 || y > h - 1.0 {
        return 0;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(w - 1.0);
    let y1 = (y0 + 1.0).min(h - 1.0);
    let dx = x - x0;
    let dy = y - y0;
    let i00 = (y0 as usize) * stride + x0 as usize;
    let i10 = (y0 as usize) * stride + x1 as usize;
    let i01 = (y1 as usize) * stride + x0 as usize;
    let i11 = (y1 as usize) * stride + x1 as usize;
    let v00 = raw[i00] as f32;
    let v10 = raw[i10] as f32;
    let v01 = raw[i01] as f32;
    let v11 = raw[i11] as f32;
    let top = v00 + (v10 - v00) * dx;
    let bot = v01 + (v11 - v01) * dx;
    (top + (bot - top) * dy).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::oriented_boxes_from_contour;

    /// Build a closed polygon contour by sampling each edge at 1-pixel steps. Mimics what
    /// `imageproc::contours::find_contours` returns for a mask of the given outline shape.
    fn sample_edges(corners: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for i in 0..corners.len() {
            let (x0, y0) = corners[i];
            let (x1, y1) = corners[(i + 1) % corners.len()];
            let dx = x1 - x0;
            let dy = y1 - y0;
            let steps = dx.abs().max(dy.abs()).round() as usize;
            for s in 0..steps {
                let t = s as f32 / steps.max(1) as f32;
                out.push((x0 + dx * t, y0 + dy * t));
            }
        }
        out
    }

    #[test]
    fn tilt_estimate_axis_aligns_word_with_uneven_ascenders() {
        // "Menu"-shaped contour: tall M on the left (cap-height), short enu on the right
        // (x-height), flat baseline. PCA on all points reports a downward-right tilt because
        // the top edge slopes down across the word; the new estimator must reject this and
        // return ~0° because the bottom edge (baseline) is flat.
        let contour = sample_edges(&[
            (0.0, 0.0),   // top-left of M cap
            (20.0, 0.0),  // top-right of M cap
            (20.0, 10.0), // step down to enu x-height
            (80.0, 10.0), // top-right of u
            (80.0, 30.0), // baseline right
            (0.0, 30.0),  // baseline left
        ]);
        let rect = oriented_boxes_from_contour(&contour).expect("rect").tight;
        let angle_deg = rect.angle_radians.to_degrees().abs();
        assert!(
            angle_deg < 1.0,
            "expected near-horizontal angle for asymmetric word mask, got {:.2}°",
            angle_deg,
        );
    }

    #[test]
    fn tilt_estimate_recovers_real_skew_when_top_and_bottom_agree() {
        // A flat horizontal rectangle (0,0)–(80,12) sheared so both edges slope down by 8 px
        // across the 80 px width — about 5.7°. Both top and bottom slope the same, so the
        // estimator should accept the tilt.
        let contour = sample_edges(&[(0.0, 0.0), (80.0, 8.0), (80.0, 20.0), (0.0, 12.0)]);
        let rect = oriented_boxes_from_contour(&contour).expect("rect").tight;
        let angle_deg = rect.angle_radians.to_degrees();
        assert!(
            (angle_deg - 5.7).abs() < 1.0,
            "expected ~5.7° tilt for genuinely sheared rectangle, got {:.2}°",
            angle_deg,
        );
    }

    #[test]
    fn tilt_estimate_axis_aligns_short_word_with_coincident_top_and_bottom_bias() {
        // Real failure from the cyberseceurope.com "Menu" label: short 4-letter word, the
        // cap-height M biases the top regression down, and AA / DBNet mask asymmetry biases
        // the bottom regression by a similar amount in the same direction. Top/bot
        // disagreement is tiny so the agreement gate accepted a ~2° phantom tilt. Sub-visible
        // deviation rejection catches it because slope × width is only ~14% of height — well
        // below the 20% visibility floor.
        let contour = sample_edges(&[
            (0.0, 0.0),  // top-left of M cap
            (16.0, 0.0), // top-right of M cap
            (16.0, 4.0), // step down to enu x-height
            (66.0, 5.0), // very slight bottom drift on top edge of enu (DBNet artefact)
            (66.0, 18.0),
            (0.0, 16.0), // very slight bottom drift on baseline (DBNet/AA artefact)
        ]);
        let rect = oriented_boxes_from_contour(&contour).expect("rect").tight;
        let angle_deg = rect.angle_radians.to_degrees().abs();
        assert!(
            angle_deg < 0.1,
            "short word with coincident top/bot bias should axis-align, got {:.2}°",
            angle_deg,
        );
    }

    #[test]
    fn tilt_estimate_recovers_small_tilt_on_long_line_with_descender_noise() {
        // Mimics one row of the multilingual packaging label: a long contour (~600 px wide,
        // ~20 px tall) tilted by ~1° (slope ≈ 0.018) with a single descender protruding 4 px
        // below the baseline. A median-y outlier filter would have rejected the tilted
        // endpoints because they sit *farthest* from the centre y, collapsing the estimate
        // to 0°. Residual-based filtering keeps the endpoints because they sit *on* the
        // fitted line, and drops only the descender.
        let tilt = 0.018f32; // ≈ 1.03°
        let mut contour = sample_edges(&[
            (0.0, 0.0),
            (600.0, 600.0 * tilt),
            (600.0, 20.0 + 600.0 * tilt),
            (350.0, 20.0 + 350.0 * tilt),
            (350.0, 24.0 + 350.0 * tilt), // descender drops 4 px below baseline
            (320.0, 24.0 + 320.0 * tilt),
            (320.0, 20.0 + 320.0 * tilt),
            (0.0, 20.0),
        ]);
        // Padding to keep enough points in the dense regions for binning.
        contour.extend(sample_edges(&[
            (10.0, 10.0 + 10.0 * tilt),
            (590.0, 10.0 + 590.0 * tilt),
        ]));
        let rect = oriented_boxes_from_contour(&contour).expect("rect").tight;
        let angle_deg = rect.angle_radians.to_degrees();
        assert!(
            (angle_deg - 1.03).abs() < 0.5,
            "expected ~1° tilt for long sloped line with one descender, got {:.2}°",
            angle_deg,
        );
    }

    #[test]
    fn tilt_estimate_ignores_descender_on_otherwise_flat_baseline() {
        // Like "page": flat top + baseline, with a single descender protrusion in one bin.
        // Outlier filtering on the bottom edge should drop the descender so the regression
        // sees a flat baseline and the angle stays ~0°.
        let contour = sample_edges(&[
            (0.0, 0.0),
            (80.0, 0.0),
            (80.0, 12.0),
            (50.0, 12.0),
            (50.0, 16.0), // descender drops 4 px below baseline
            (40.0, 16.0),
            (40.0, 12.0),
            (0.0, 12.0),
        ]);
        let rect = oriented_boxes_from_contour(&contour).expect("rect").tight;
        let angle_deg = rect.angle_radians.to_degrees().abs();
        assert!(
            angle_deg < 1.0,
            "expected near-horizontal angle for word with one descender, got {:.2}°",
            angle_deg,
        );
    }
}

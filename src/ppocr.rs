use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use image::{DynamicImage, GenericImageView, GrayImage, RgbImage, imageops::FilterType};
use imageproc::contours::find_contours;
use imageproc::point::Point;
use rayon::ThreadPool;
use rayon::prelude::*;

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::mnn_inference::MnnSession;

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

const PPOCR_DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const PPOCR_DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const PPOCR_REC_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const PPOCR_REC_STD: [f32; 3] = [0.5, 0.5, 0.5];

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

#[derive(Debug, Clone)]
pub struct PpocrLine {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: PpocrRect,
    /// Min-area-aligned rotated rectangle around the detection contour. The principal axis is
    /// the text's reading direction, so for tilted signs/paper this stays tight to the glyphs
    /// instead of inflating to an axis-aligned bounding box like `bounding_box` does.
    pub oriented_box: crate::ocr::OrientedRect,
    /// Pre-inflate min-area rect from the raw DB mask contour — same centre/angle as
    /// `oriented_box` but without the unclip/border padding. Tight to the segmentation kernel,
    /// so its height excludes most ascender/descender whitespace, making it the right metric
    /// for paragraph grouping (line-height clustering, gap-as-multiple-of-x-height). For
    /// detections that come back without a contour (axis-aligned fallback), this equals
    /// `oriented_box`.
    pub tight_box: crate::ocr::OrientedRect,
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

pub struct PpocrEngine {
    detector: PpocrDetector,
    recognizer: PpocrRecognizer,
}

impl PpocrEngine {
    pub fn load(
        det_path: &Path,
        rec_path: &Path,
        keys_path: &Path,
        det_intra_threads: usize,
    ) -> Result<Self, TranslatorError> {
        // Det is one big graph and benefits from intra-session threading. Rec uses a
        // single-threaded session pool dispatched in parallel — see PpocrRecognizer::load.
        let detector = PpocrDetector::load(det_path, det_intra_threads)?;
        let recognizer = PpocrRecognizer::load(rec_path, keys_path)?;
        Ok(Self {
            detector,
            recognizer,
        })
    }

    pub fn recognize_rgba(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<PpocrLine>, TranslatorError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                TranslatorError::new(TranslatorErrorKind::InvalidInput, "image dims overflow")
            })?;
        if rgba.len() != expected {
            return Err(TranslatorError::new(
                TranslatorErrorKind::InvalidInput,
                format!(
                    "rgba length {} != {}x{}x4 ({})",
                    rgba.len(),
                    width,
                    height,
                    expected
                ),
            ));
        }

        let t_rgba = Instant::now();
        let image = rgba_to_dynamic(rgba, width, height);
        let rgba_ms = t_rgba.elapsed().as_secs_f32() * 1000.0;

        let boxes = self.detector.detect(&image)?;
        log::info!(
            "ppocr: detector returned {} boxes ({}x{})",
            boxes.len(),
            width,
            height
        );
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let t_crops = Instant::now();
        let gray = image.to_luma8();
        let mut crops = Vec::with_capacity(boxes.len());
        let mut box_meta = Vec::with_capacity(boxes.len());
        let mut oriented_meta: Vec<Option<ContourBoxes>> = Vec::with_capacity(boxes.len());
        for tb in boxes {
            let expanded = expand_box(&tb.rect, DET_BOX_BORDER, width, height);
            let oriented = tb
                .contour
                .as_ref()
                .and_then(|c| oriented_boxes_from_contour(c));
            let crop_image = tb
                .contour
                .as_ref()
                .and_then(|c| dewarp_contour_to_strip(&gray, c))
                .map(DynamicImage::ImageLuma8)
                .unwrap_or_else(|| crop_dynamic(&image, &expanded));
            crops.push(crop_image);
            box_meta.push(expanded);
            oriented_meta.push(oriented);
        }
        let crops_ms = t_crops.elapsed().as_secs_f32() * 1000.0;

        let n_crops = crops.len();
        let timings_us = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
        let recognizer = &self.recognizer;
        let t_rec_wall = Instant::now();
        let results: Vec<Result<RecResult, TranslatorError>> = recognizer.pool.install(|| {
            crops
                .par_iter()
                .map(|crop| {
                    let worker_idx =
                        rayon::current_thread_index().unwrap_or(0) % recognizer.sessions.len();
                    recognizer.recognize_one(crop, worker_idx, &timings_us)
                })
                .collect()
        });
        let rec_wall_ms = t_rec_wall.elapsed().as_secs_f32() * 1000.0;
        let rec_pre_ms = timings_us.0.load(Ordering::Relaxed) as f32 / 1000.0;
        let rec_infer_ms = timings_us.1.load(Ordering::Relaxed) as f32 / 1000.0;
        let rec_post_ms = timings_us.2.load(Ordering::Relaxed) as f32 / 1000.0;

        let mut lines = Vec::with_capacity(n_crops);
        let mut empty_count = 0usize;
        let mut low_score_count = 0usize;
        for (index, result) in results.into_iter().enumerate() {
            let result = result?;
            if result.text.trim().is_empty() {
                empty_count += 1;
                continue;
            }
            if result.confidence < REC_DROP_SCORE {
                low_score_count += 1;
                continue;
            }
            {
                let (oriented, tight) = match oriented_meta[index] {
                    Some(ContourBoxes { tight, inflated }) => (inflated, tight),
                    None => {
                        let aabb = crate::ocr::OrientedRect::axis_aligned(crate::ocr::Rect {
                            left: box_meta[index].left,
                            top: box_meta[index].top,
                            right: box_meta[index].right,
                            bottom: box_meta[index].bottom,
                        });
                        (aabb, aabb)
                    }
                };
                lines.push(PpocrLine {
                    text: result.text,
                    confidence: result.confidence,
                    bounding_box: box_meta[index],
                    oriented_box: oriented,
                    tight_box: tight,
                });
            }
        }
        log::info!(
            "ppocr: {}/{} regions recognized ({} empty, {} below drop_score {:.2}) — \
             rgba_pack={:.1}ms crops/dewarp={:.1}ms \
             rec_wall={:.1}ms (cpu pre={:.1}ms infer={:.1}ms post={:.1}ms over {} workers)",
            lines.len(),
            n_crops,
            empty_count,
            low_score_count,
            REC_DROP_SCORE,
            rgba_ms,
            crops_ms,
            rec_wall_ms,
            rec_pre_ms,
            rec_infer_ms,
            rec_post_ms,
            REC_PARALLELISM,
        );
        sort_lines_reading_order(&mut lines);
        Ok(lines)
    }
}

fn rgba_to_dynamic(rgba: &[u8], width: u32, height: u32) -> DynamicImage {
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

fn expand_box(rect: &PpocrRect, border: u32, max_w: u32, max_h: u32) -> PpocrRect {
    PpocrRect {
        left: rect.left.saturating_sub(border),
        top: rect.top.saturating_sub(border),
        right: (rect.right + border).min(max_w),
        bottom: (rect.bottom + border).min(max_h),
    }
}

fn sort_lines_reading_order(lines: &mut [PpocrLine]) {
    if lines.is_empty() {
        return;
    }
    // Bucket by `top / (median_height / 2)` so lines that share a row land in the same bucket
    // regardless of small vertical jitter. Using a global bucket size (rather than per-pair) is
    // what keeps the comparator a true total order — the previous per-pair `max(height_a,
    // height_b)` approach made `same_row` non-transitive and panicked under Rust's tightened
    // sort checks.
    let mut heights: Vec<u32> = lines
        .iter()
        .map(|l| l.bounding_box.height().max(1))
        .collect();
    heights.sort_unstable();
    let median_h = heights[heights.len() / 2].max(1);
    let bucket = (median_h / 2).max(1);
    lines.sort_by_key(|l| (l.bounding_box.top / bucket, l.bounding_box.left));
}

// ---------- Detector ----------

#[derive(Debug, Clone)]
struct DetBox {
    rect: PpocrRect,
    contour: Option<Vec<(f32, f32)>>,
}

impl PpocrDetector {
    fn load(model_path: &Path, intra_threads: usize) -> Result<Self, TranslatorError> {
        let session = MnnSession::load(model_path, intra_threads)?;
        Ok(Self { session })
    }

    fn detect(&self, image: &DynamicImage) -> Result<Vec<DetBox>, TranslatorError> {
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
            if v > DET_SCORE_THRESHOLD {
                over_thresh += 1;
            }
        }
        let mask_mean = mask_sum / mask.len() as f32;
        let t_post = Instant::now();
        let binary: Vec<u8> = mask
            .iter()
            .map(|&v| if v > DET_SCORE_THRESHOLD { 255 } else { 0 })
            .collect();
        let boxes = extract_boxes(
            &binary, &mask, out_w, out_h, scaled_w, scaled_h, orig_w, orig_h,
        );
        let post_ms = t_post.elapsed().as_secs_f32() * 1000.0;
        log::info!(
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
            DET_SCORE_THRESHOLD,
            over_thresh,
            mask.len(),
            pre_ms,
            infer_ms,
            post_ms,
        );
        Ok(boxes)
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

fn extract_boxes(
    mask: &[u8],
    heatmap: &[f32],
    mask_w: u32,
    mask_h: u32,
    valid_w: u32,
    valid_h: u32,
    orig_w: u32,
    orig_h: u32,
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
        if box_w * box_h < DET_MIN_AREA {
            continue;
        }

        // Box-score gate (PaddleOCR's `det_db_box_thresh`, "fast" score_mode): mean heatmap
        // probability over the contour's AABB on the *raw* float output. Real text masks
        // average 0.7+ here; texture/compression noise that just barely cleared the per-pixel
        // threshold rarely makes it past 0.5.
        let mut score_sum = 0.0f32;
        let mut score_n = 0usize;
        for y in (min_y as u32)..(max_y as u32).min(mask_h) {
            let row = (y as usize) * (mask_w as usize);
            for x in (min_x as u32)..(max_x as u32).min(mask_w) {
                score_sum += heatmap[row + x as usize];
                score_n += 1;
            }
        }
        let box_score = if score_n > 0 {
            score_sum / score_n as f32
        } else {
            0.0
        };
        if box_score < DET_BOX_MIN_SCORE {
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
        });
    }
    if weak_score_count > 0 {
        log::info!(
            "ppocr det: {} contour(s) below box-score gate {:.2}",
            weak_score_count,
            DET_BOX_MIN_SCORE,
        );
    }
    boxes
}

// ---------- Recognizer ----------

struct RecResult {
    text: String,
    confidence: f32,
}

struct RecTiming {
    pre_ms: f32,
    infer_ms: f32,
    post_ms: f32,
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
    // Canonicalize tangent to point in +x so angle ∈ (-π/2, π/2]. Without this, lines tilted
    // counterclockwise from horizontal could come back with the tangent pointing leftward and
    // an angle near ±π — the renderer would draw the text backwards.
    if ux < 0.0 {
        ux = -ux;
        uy = -uy;
    }
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
    let angle_radians = uy.atan2(ux);

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

use std::path::Path;
use std::sync::Mutex;

use image::{DynamicImage, GenericImageView, GrayImage, RgbImage, imageops::FilterType};
use imageproc::contours::find_contours;
use imageproc::point::Point;
use ndarray::Array4;
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::inference::load_onnx_session;

const DET_INPUT_NAME: &str = "x";
const REC_INPUT_NAME: &str = "x";
const REC_TARGET_HEIGHT: u32 = 48;
const REC_MIN_SCORE: f32 = 0.3;
const REC_PUNCT_MIN_SCORE: f32 = 0.1;
const REC_BATCH_SIZE: usize = 16;
// Hard OOM ceiling, not a perf cap. The user's `maxImageSize` setting (and
// the doc-align warp step) already determine the working resolution; we only
// step in if something pathological reaches us.
const DET_MAX_SIDE: u32 = 4096;
const DET_SCORE_THRESHOLD: f32 = 0.3;
const DET_MIN_AREA: u32 = 16;
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
}

pub struct PpocrDetector {
    session: Mutex<Session>,
}

pub struct PpocrRecognizer {
    session: Mutex<Session>,
    charset: Vec<char>,
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
        intra_threads: usize,
    ) -> Result<Self, TranslatorError> {
        let detector = PpocrDetector::load(det_path, intra_threads)?;
        let recognizer = PpocrRecognizer::load(rec_path, keys_path, intra_threads)?;
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

        let image = rgba_to_dynamic(rgba, width, height);
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

        let gray = image.to_luma8();
        let mut crops = Vec::with_capacity(boxes.len());
        let mut box_meta = Vec::with_capacity(boxes.len());
        for tb in boxes {
            let expanded = expand_box(&tb.rect, DET_BOX_BORDER, width, height);
            let crop_image = tb
                .contour
                .as_ref()
                .and_then(|c| dewarp_contour_to_strip(&gray, c))
                .map(DynamicImage::ImageLuma8)
                .unwrap_or_else(|| crop_dynamic(&image, &expanded));
            crops.push(crop_image);
            box_meta.push(expanded);
        }

        let n_crops = crops.len();
        let mut lines = Vec::with_capacity(n_crops);
        let mut empty_count = 0usize;
        for chunk_start in (0..n_crops).step_by(REC_BATCH_SIZE) {
            let end = (chunk_start + REC_BATCH_SIZE).min(n_crops);
            let chunk = &crops[chunk_start..end];
            let results = self.recognizer.recognize_batch(chunk)?;
            for (offset, result) in results.into_iter().enumerate() {
                if result.text.trim().is_empty() {
                    empty_count += 1;
                    continue;
                }
                lines.push(PpocrLine {
                    text: result.text,
                    confidence: result.confidence,
                    bounding_box: box_meta[chunk_start + offset],
                });
            }
        }
        log::info!(
            "ppocr: {}/{} regions recognized ({} empty)",
            lines.len(),
            n_crops,
            empty_count
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
    lines.sort_by(|a, b| {
        let ay = a.bounding_box.top;
        let by = b.bounding_box.top;
        let row_height = a.bounding_box.height().max(b.bounding_box.height()).max(1);
        let same_row = ay.abs_diff(by) <= row_height / 2;
        if same_row {
            a.bounding_box.left.cmp(&b.bounding_box.left)
        } else {
            ay.cmp(&by)
        }
    });
}

// ---------- Detector ----------

#[derive(Debug, Clone)]
struct DetBox {
    rect: PpocrRect,
    contour: Option<Vec<(f32, f32)>>,
}

impl PpocrDetector {
    fn load(model_path: &Path, intra_threads: usize) -> Result<Self, TranslatorError> {
        let session = load_onnx_session(model_path, intra_threads)?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    fn detect(&self, image: &DynamicImage) -> Result<Vec<DetBox>, TranslatorError> {
        let (orig_w, orig_h) = image.dimensions();
        let scaled = resize_to_max_side(image, DET_MAX_SIDE);
        let (scaled_w, scaled_h) = scaled.dimensions();
        let pad_w = pad_to_multiple(scaled_w, 32);
        let pad_h = pad_to_multiple(scaled_h, 32);
        let tensor_buf = preprocess_for_det(&scaled, pad_w, pad_h);

        let input = Tensor::from_array((
            [1usize, 3, pad_h as usize, pad_w as usize],
            tensor_buf.into_boxed_slice(),
        ))
        .map_err(|e| {
            TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!("ppocr det tensor build failed: {e}"),
            )
        })?;

        let (out_shape, mask) = {
            let mut session = self.session.lock().expect("ppocr det session poisoned");
            let outputs = session.run(inputs![DET_INPUT_NAME => input]).map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("ppocr det inference failed: {e}"),
                )
            })?;
            let (shape, mask) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("ppocr det output not f32: {e}"),
                )
            })?;
            (shape.to_vec(), mask.to_vec())
        };

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
        log::info!(
            "ppocr det: input_pad={}x{} scaled={}x{} out={}x{} mask[min/max/mean]={:.3}/{:.3}/{:.3} over_{}={}/{}",
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
            mask.len()
        );
        let binary: Vec<u8> = mask
            .iter()
            .map(|&v| if v > DET_SCORE_THRESHOLD { 255 } else { 0 })
            .collect();

        Ok(extract_boxes(
            &binary, out_w, out_h, scaled_w, scaled_h, orig_w, orig_h,
        ))
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
    boxes
}

// ---------- Recognizer ----------

struct RecResult {
    text: String,
    confidence: f32,
}

impl PpocrRecognizer {
    fn load(
        model_path: &Path,
        keys_path: &Path,
        intra_threads: usize,
    ) -> Result<Self, TranslatorError> {
        let session = load_onnx_session(model_path, intra_threads)?;
        let charset = load_charset(keys_path)?;
        Ok(Self {
            session: Mutex::new(session),
            charset,
        })
    }

    fn recognize_batch(&self, images: &[DynamicImage]) -> Result<Vec<RecResult>, TranslatorError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let target_h = REC_TARGET_HEIGHT as usize;
        let scaled_widths: Vec<u32> = images
            .iter()
            .map(|img| {
                let (w, h) = img.dimensions();
                let scale = REC_TARGET_HEIGHT as f32 / h as f32;
                (w as f32 * scale).round().max(1.0) as u32
            })
            .collect();
        let max_w = *scaled_widths.iter().max().unwrap() as usize;
        let n = images.len();
        let mut batch = Array4::<f32>::zeros((n, 3, target_h, max_w));
        for (i, (img, &sw)) in images.iter().zip(scaled_widths.iter()).enumerate() {
            let resized = img.resize_exact(sw, REC_TARGET_HEIGHT, FilterType::Triangle);
            let rgb = resized.to_rgb8();
            for y in 0..target_h {
                for x in 0..sw as usize {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    batch[[i, 0, y, x]] =
                        (pixel[0] as f32 / 255.0 - PPOCR_REC_MEAN[0]) / PPOCR_REC_STD[0];
                    batch[[i, 1, y, x]] =
                        (pixel[1] as f32 / 255.0 - PPOCR_REC_MEAN[1]) / PPOCR_REC_STD[1];
                    batch[[i, 2, y, x]] =
                        (pixel[2] as f32 / 255.0 - PPOCR_REC_MEAN[2]) / PPOCR_REC_STD[2];
                }
            }
        }
        let (input_data, _offset) = batch.into_raw_vec_and_offset();
        let input = Tensor::from_array(([n, 3, target_h, max_w], input_data.into_boxed_slice()))
            .map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("ppocr rec tensor build failed: {e}"),
                )
            })?;

        let (out_shape, out_data) = {
            let mut session = self.session.lock().expect("ppocr rec session poisoned");
            let outputs = session.run(inputs![REC_INPUT_NAME => input]).map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("ppocr rec inference failed: {e}"),
                )
            })?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("ppocr rec output not f32: {e}"),
                )
            })?;
            (shape.to_vec(), data.to_vec())
        };

        if out_shape.len() != 3 {
            return Err(TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!("ppocr rec output shape unexpected: {:?}", out_shape),
            ));
        }
        let batch_n = out_shape[0] as usize;
        let seq_len = out_shape[1] as usize;
        let num_classes = out_shape[2] as usize;
        let mut results = Vec::with_capacity(batch_n);
        for b in 0..batch_n {
            let sample_offset = b * seq_len * num_classes;
            results.push(decode_ctc(
                &out_data[sample_offset..sample_offset + seq_len * num_classes],
                seq_len,
                num_classes,
                &self.charset,
            ));
        }
        Ok(results)
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

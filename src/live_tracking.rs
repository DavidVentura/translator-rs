//! Lightweight live-camera motion tracking for OCR overlays.
//!
//! The tracker estimates frame-to-frame crop translation from downscaled
//! grayscale images. It deliberately returns only global motion: this is stable
//! for the live OCR use case where the user points at one dominant sign/package
//! and detector observations are used periodically to correct drift.

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::ocr::Rect;

const DEFAULT_TARGET_PIXELS: u32 = 96_000;
const MIN_DIMENSION: u32 = 48;
const PATCH_RADIUS: i32 = 4;
const GRID_COLS: u32 = 9;
const GRID_ROWS: u32 = 7;
const MIN_TEXTURE: f32 = 16.0;
const MAX_MEAN_ABS_DIFF: f32 = 42.0;
const INLIER_RADIUS_PX: f32 = 3.0;
const MIN_INLIERS: usize = 8;

#[derive(Debug, Clone, Copy, Default)]
pub struct LiveMotionEstimate {
    pub valid: bool,
    pub dx: f32,
    pub dy: f32,
    pub confidence: f32,
    pub matches: u32,
    pub inliers: u32,
    pub reset: bool,
}

#[derive(Debug, Clone)]
struct TrackingImage {
    gray: Vec<u8>,
    width: u32,
    height: u32,
    scale_to_full_x: f32,
    scale_to_full_y: f32,
    crop: Rect,
}

#[derive(Debug, Default)]
pub struct LiveFrameTracker {
    previous: Option<TrackingImage>,
}

impl LiveFrameTracker {
    pub fn new() -> Self {
        Self { previous: None }
    }

    pub fn reset(&mut self) {
        self.previous = None;
    }

    pub fn update(
        &mut self,
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
    ) -> Result<LiveMotionEstimate, TranslatorError> {
        self.update_with_target_pixels(
            rgba,
            sensor_width,
            sensor_height,
            rotation_degrees,
            display_crop,
            DEFAULT_TARGET_PIXELS,
        )
    }

    pub fn update_with_target_pixels(
        &mut self,
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
        target_pixels: u32,
    ) -> Result<LiveMotionEstimate, TranslatorError> {
        let current = build_tracking_image(
            rgba,
            sensor_width,
            sensor_height,
            rotation_degrees,
            display_crop,
            target_pixels,
        )?;
        let Some(previous) = self.previous.as_ref() else {
            self.previous = Some(current);
            return Ok(LiveMotionEstimate {
                reset: true,
                ..LiveMotionEstimate::default()
            });
        };

        let compatible = previous.width == current.width
            && previous.height == current.height
            && previous.crop.width() == current.crop.width()
            && previous.crop.height() == current.crop.height();
        if !compatible {
            self.previous = Some(current);
            return Ok(LiveMotionEstimate {
                reset: true,
                ..LiveMotionEstimate::default()
            });
        }

        let estimate = estimate_translation(previous, &current);
        self.previous = Some(current);
        Ok(estimate)
    }
}

fn build_tracking_image(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
    rotation_degrees: i32,
    display_crop: Rect,
    target_pixels: u32,
) -> Result<TrackingImage, TranslatorError> {
    let expected = (sensor_width as usize)
        .checked_mul(sensor_height as usize)
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
                sensor_width,
                sensor_height,
                expected,
            ),
        ));
    }
    if display_crop.right <= display_crop.left || display_crop.bottom <= display_crop.top {
        return Err(TranslatorError::new(
            TranslatorErrorKind::InvalidInput,
            "tracking crop is empty",
        ));
    }

    let crop_w = display_crop.width();
    let crop_h = display_crop.height();
    let crop_pixels = (crop_w as u64) * (crop_h as u64);
    let target = target_pixels.max(1) as u64;
    let scale = if crop_pixels > target {
        (target as f64 / crop_pixels as f64).sqrt() as f32
    } else {
        1.0
    };
    let out_w = ((crop_w as f32) * scale).round().max(MIN_DIMENSION as f32) as u32;
    let out_h = ((crop_h as f32) * scale).round().max(MIN_DIMENSION as f32) as u32;
    let out_w = out_w.min(crop_w.max(1));
    let out_h = out_h.min(crop_h.max(1));
    let scale_to_full_x = crop_w as f32 / out_w as f32;
    let scale_to_full_y = crop_h as f32 / out_h as f32;

    let mut gray = vec![0u8; (out_w as usize) * (out_h as usize)];
    for oy in 0..out_h {
        let display_y = display_crop.top as f32 + (oy as f32 + 0.5) * scale_to_full_y;
        for ox in 0..out_w {
            let display_x = display_crop.left as f32 + (ox as f32 + 0.5) * scale_to_full_x;
            let (sx, sy) = display_to_sensor(
                display_x,
                display_y,
                sensor_width,
                sensor_height,
                rotation_degrees,
            )?;
            let idx = ((sy * sensor_width + sx) * 4) as usize;
            let r = rgba[idx] as f32;
            let g = rgba[idx + 1] as f32;
            let b = rgba[idx + 2] as f32;
            gray[(oy * out_w + ox) as usize] = (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    Ok(TrackingImage {
        gray,
        width: out_w,
        height: out_h,
        scale_to_full_x,
        scale_to_full_y,
        crop: display_crop,
    })
}

fn display_to_sensor(
    x: f32,
    y: f32,
    sensor_w: u32,
    sensor_h: u32,
    rotation_degrees: i32,
) -> Result<(u32, u32), TranslatorError> {
    let r = ((rotation_degrees % 360) + 360) % 360;
    let (sx, sy) = match r {
        0 => (x, y),
        90 => (y, sensor_h as f32 - 1.0 - x),
        180 => (sensor_w as f32 - 1.0 - x, sensor_h as f32 - 1.0 - y),
        270 => (sensor_w as f32 - 1.0 - y, x),
        _ => {
            return Err(TranslatorError::new(
                TranslatorErrorKind::InvalidInput,
                format!("unsupported rotation_degrees: {}", rotation_degrees),
            ));
        }
    };
    Ok((
        sx.round().clamp(0.0, sensor_w.saturating_sub(1) as f32) as u32,
        sy.round().clamp(0.0, sensor_h.saturating_sub(1) as f32) as u32,
    ))
}

fn estimate_translation(previous: &TrackingImage, current: &TrackingImage) -> LiveMotionEstimate {
    let w = previous.width as i32;
    let h = previous.height as i32;
    let search_radius = ((w.min(h) as f32) * 0.055).round().clamp(6.0, 24.0) as i32;
    let margin = PATCH_RADIUS + search_radius + 1;
    if w <= margin * 2 || h <= margin * 2 {
        return LiveMotionEstimate::default();
    }

    let mut matches = Vec::with_capacity((GRID_COLS * GRID_ROWS) as usize);
    for gy in 0..GRID_ROWS {
        let y = margin + (((h - 2 * margin) as f32) * (gy as f32 + 0.5) / GRID_ROWS as f32) as i32;
        for gx in 0..GRID_COLS {
            let x =
                margin + (((w - 2 * margin) as f32) * (gx as f32 + 0.5) / GRID_COLS as f32) as i32;
            if patch_texture(previous, x, y) < MIN_TEXTURE {
                continue;
            }
            if let Some(m) = match_patch(previous, current, x, y, search_radius) {
                matches.push(m);
            }
        }
    }

    if matches.len() < MIN_INLIERS {
        return LiveMotionEstimate {
            matches: matches.len() as u32,
            ..LiveMotionEstimate::default()
        };
    }

    let mut dxs: Vec<f32> = matches.iter().map(|m| m.dx as f32).collect();
    let mut dys: Vec<f32> = matches.iter().map(|m| m.dy as f32).collect();
    let med_dx = median(&mut dxs);
    let med_dy = median(&mut dys);

    let mut inliers = Vec::with_capacity(matches.len());
    for m in &matches {
        let ddx = m.dx as f32 - med_dx;
        let ddy = m.dy as f32 - med_dy;
        if (ddx * ddx + ddy * ddy).sqrt() <= INLIER_RADIUS_PX {
            inliers.push(*m);
        }
    }
    if inliers.len() < MIN_INLIERS {
        return LiveMotionEstimate {
            matches: matches.len() as u32,
            inliers: inliers.len() as u32,
            ..LiveMotionEstimate::default()
        };
    }

    let dx = inliers.iter().map(|m| m.dx as f32).sum::<f32>() / inliers.len() as f32;
    let dy = inliers.iter().map(|m| m.dy as f32).sum::<f32>() / inliers.len() as f32;
    let mean_error = inliers.iter().map(|m| m.mean_abs_diff).sum::<f32>() / inliers.len() as f32;
    let inlier_ratio = inliers.len() as f32 / matches.len() as f32;
    let error_score = (1.0 - mean_error / MAX_MEAN_ABS_DIFF).clamp(0.0, 1.0);
    let confidence = (inlier_ratio * 0.75 + error_score * 0.25).clamp(0.0, 1.0);
    if confidence < 0.35 {
        return LiveMotionEstimate {
            matches: matches.len() as u32,
            inliers: inliers.len() as u32,
            confidence,
            ..LiveMotionEstimate::default()
        };
    }

    LiveMotionEstimate {
        valid: true,
        dx: dx * current.scale_to_full_x,
        dy: dy * current.scale_to_full_y,
        confidence,
        matches: matches.len() as u32,
        inliers: inliers.len() as u32,
        reset: false,
    }
}

#[derive(Debug, Clone, Copy)]
struct PatchMatch {
    dx: i32,
    dy: i32,
    mean_abs_diff: f32,
}

fn patch_texture(image: &TrackingImage, cx: i32, cy: i32) -> f32 {
    let mut min_v = u8::MAX;
    let mut max_v = u8::MIN;
    for y in (cy - PATCH_RADIUS)..=(cy + PATCH_RADIUS) {
        for x in (cx - PATCH_RADIUS)..=(cx + PATCH_RADIUS) {
            let v = image.gray[(y as u32 * image.width + x as u32) as usize];
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    (max_v - min_v) as f32
}

fn match_patch(
    previous: &TrackingImage,
    current: &TrackingImage,
    x: i32,
    y: i32,
    search_radius: i32,
) -> Option<PatchMatch> {
    let mut best_sad = u32::MAX;
    let mut best_dx = 0;
    let mut best_dy = 0;
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let sad = patch_sad(previous, current, x, y, dx, dy);
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    let pixels = ((PATCH_RADIUS * 2 + 1) * (PATCH_RADIUS * 2 + 1)) as f32;
    let mean_abs_diff = best_sad as f32 / pixels;
    (mean_abs_diff <= MAX_MEAN_ABS_DIFF).then_some(PatchMatch {
        dx: best_dx,
        dy: best_dy,
        mean_abs_diff,
    })
}

fn patch_sad(
    previous: &TrackingImage,
    current: &TrackingImage,
    cx: i32,
    cy: i32,
    dx: i32,
    dy: i32,
) -> u32 {
    let mut sad = 0u32;
    for py in -PATCH_RADIUS..=PATCH_RADIUS {
        let prev_row = ((cy + py) as u32 * previous.width) as usize;
        let curr_row = ((cy + dy + py) as u32 * current.width) as usize;
        for px in -PATCH_RADIUS..=PATCH_RADIUS {
            let a = previous.gray[prev_row + (cx + px) as usize] as i32;
            let b = current.gray[curr_row + (cx + dx + px) as usize] as i32;
            sad += a.abs_diff(b);
        }
    }
    sad
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

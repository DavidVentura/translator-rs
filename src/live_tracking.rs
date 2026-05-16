//! Lightweight live-camera motion tracking for OCR overlays.
//!
//! The tracker estimates frame-to-frame crop translation from downscaled
//! grayscale images. It deliberately returns only global motion: this is stable
//! for the live OCR use case where the user points at one dominant sign/package
//! and detector observations are used periodically to correct drift.

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::ocr::Rect;

const DEFAULT_TARGET_PIXELS: u32 = 96_000;

/// Tracking-image pixel budget used by `update()`. Exposed so binding code that
/// calls `update_with_regions` can pass the same default.
pub const fn default_target_pixels() -> u32 {
    DEFAULT_TARGET_PIXELS
}
const MIN_DIMENSION: u32 = 48;
const PATCH_RADIUS: i32 = 4;
const GRID_COLS: u32 = 9;
const GRID_ROWS: u32 = 7;
const MIN_TEXTURE: f32 = 16.0;
const MAX_MEAN_ABS_DIFF: f32 = 42.0;
const INLIER_RADIUS_PX: f32 = 3.0;
const MIN_INLIERS: usize = 8;

const INVALID_MOTION_PRIOR_DECAY: f32 = 0.5;

const REGION_GRID_COLS: u32 = 5;
const REGION_GRID_ROWS: u32 = 5;
const REGION_LOCAL_SEARCH_PX: i32 = 5;
const REGION_MIN_INLIERS: usize = 6;
const REGION_INLIER_RESIDUAL_PX: f32 = 2.0;

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

/// Per-region similarity (4-DOF: translation + uniform scale + rotation),
/// returned as a 2x3 affine matrix in display-coord space:
///   new_x = a * old_x + b * old_y + c
///   new_y = d * old_x + e * old_y + f
/// For a similarity, e == a and d == -b; the struct just stores the full
/// six coefficients so callers don't need to know which model produced them.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiveRegionMotion {
    pub valid: bool,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
    pub inliers: u32,
    pub matches: u32,
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
    /// Last successful global motion, in *tracking-image* pixels. Used as a
    /// search prior so the bounded SAD window only has to cover the change in
    /// velocity, not absolute velocity.
    prior_dx_track: f32,
    prior_dy_track: f32,
}

impl LiveFrameTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.previous = None;
        self.prior_dx_track = 0.0;
        self.prior_dy_track = 0.0;
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
        let (estimate, _) = self.update_with_regions(
            rgba,
            sensor_width,
            sensor_height,
            rotation_degrees,
            display_crop,
            target_pixels,
            &[],
        )?;
        Ok(estimate)
    }

    /// Like `update`, but additionally fits a per-region similarity for each
    /// supplied region rect (in display coords). Returns (global, per-region)
    /// with the per-region vec aligned to `regions`. Invalid entries (too few
    /// inliers etc.) come back with `valid = false`; callers should fall back
    /// to global motion for those tracks.
    pub fn update_with_regions(
        &mut self,
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
        target_pixels: u32,
        regions: &[Rect],
    ) -> Result<(LiveMotionEstimate, Vec<LiveRegionMotion>), TranslatorError> {
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
            return Ok((
                LiveMotionEstimate {
                    reset: true,
                    ..LiveMotionEstimate::default()
                },
                vec![LiveRegionMotion::default(); regions.len()],
            ));
        };

        let compatible = previous.width == current.width
            && previous.height == current.height
            && previous.crop.width() == current.crop.width()
            && previous.crop.height() == current.crop.height();
        if !compatible {
            self.previous = Some(current);
            return Ok((
                LiveMotionEstimate {
                    reset: true,
                    ..LiveMotionEstimate::default()
                },
                vec![LiveRegionMotion::default(); regions.len()],
            ));
        }

        let prior_dx = self.prior_dx_track.round() as i32;
        let prior_dy = self.prior_dy_track.round() as i32;
        let global = estimate_translation(previous, &current, prior_dx, prior_dy);
        let region_motions = if global.valid && !regions.is_empty() {
            estimate_per_region(previous, &current, regions, global.dx, global.dy)
        } else {
            vec![LiveRegionMotion::default(); regions.len()]
        };
        if global.valid && current.scale_to_full_x > 0.0 && current.scale_to_full_y > 0.0 {
            self.prior_dx_track = global.dx / current.scale_to_full_x;
            self.prior_dy_track = global.dy / current.scale_to_full_y;
        } else {
            // Decay rather than reset: if motion failed because the user is
            // mid-fast-pan and blur briefly killed SAD, the next frame is
            // probably still moving at roughly the same velocity. Halving on
            // each consecutive invalid lets us extrapolate for one or two
            // frames before the prior fades to zero (avoids overshooting if
            // the user is actually decelerating).
            self.prior_dx_track *= INVALID_MOTION_PRIOR_DECAY;
            self.prior_dy_track *= INVALID_MOTION_PRIOR_DECAY;
        }
        self.previous = Some(current);
        Ok((global, region_motions))
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

fn estimate_translation(
    previous: &TrackingImage,
    current: &TrackingImage,
    prior_dx: i32,
    prior_dy: i32,
) -> LiveMotionEstimate {
    let w = previous.width as i32;
    let h = previous.height as i32;
    let search_radius = ((w.min(h) as f32) * 0.055).round().clamp(6.0, 24.0) as i32;
    let margin = PATCH_RADIUS + search_radius + 1 + prior_dx.abs().max(prior_dy.abs());
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
            if let Some(m) = match_patch(previous, current, x, y, search_radius, prior_dx, prior_dy)
            {
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

    let mut dxs: Vec<f32> = matches.iter().map(|m| m.dx).collect();
    let mut dys: Vec<f32> = matches.iter().map(|m| m.dy).collect();
    let med_dx = median(&mut dxs);
    let med_dy = median(&mut dys);

    let mut inliers = Vec::with_capacity(matches.len());
    for m in &matches {
        let ddx = m.dx - med_dx;
        let ddy = m.dy - med_dy;
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

    let dx = inliers.iter().map(|m| m.dx).sum::<f32>() / inliers.len() as f32;
    let dy = inliers.iter().map(|m| m.dy).sum::<f32>() / inliers.len() as f32;
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
    /// Patch centre in the previous tracking image (integer pixel).
    prev_x: f32,
    prev_y: f32,
    /// Sub-pixel displacement in the current tracking image.
    dx: f32,
    dy: f32,
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

/// Search around `(x, y)` for the patch that minimises SAD with the previous
/// patch at `(x, y)`. The search is centred on `(prior_dx, prior_dy)` — pass
/// zero for global search, or the predicted displacement for a localised refine.
/// On success returns a `PatchMatch` whose `dx`/`dy` include parabolic sub-pixel
/// refinement around the integer minimum.
fn match_patch(
    previous: &TrackingImage,
    current: &TrackingImage,
    x: i32,
    y: i32,
    search_radius: i32,
    prior_dx: i32,
    prior_dy: i32,
) -> Option<PatchMatch> {
    // Reject anything that would walk off either image (the current centre
    // moves to `(x + prior_dx + dx, y + prior_dy + dy)`).
    let w = current.width as i32;
    let h = current.height as i32;
    let cx_min = (x + prior_dx) - search_radius;
    let cx_max = (x + prior_dx) + search_radius;
    let cy_min = (y + prior_dy) - search_radius;
    let cy_max = (y + prior_dy) + search_radius;
    if cx_min - PATCH_RADIUS < 0
        || cx_max + PATCH_RADIUS >= w
        || cy_min - PATCH_RADIUS < 0
        || cy_max + PATCH_RADIUS >= h
    {
        return None;
    }
    let span = (search_radius * 2 + 1) as usize;
    let mut sads = vec![u32::MAX; span * span];
    let mut best_sad = u32::MAX;
    let mut best_dx = 0;
    let mut best_dy = 0;
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let sad = patch_sad(previous, current, x, y, prior_dx + dx, prior_dy + dy);
            let idx = ((dy + search_radius) as usize) * span + (dx + search_radius) as usize;
            sads[idx] = sad;
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    let pixels = ((PATCH_RADIUS * 2 + 1) * (PATCH_RADIUS * 2 + 1)) as f32;
    let mean_abs_diff = best_sad as f32 / pixels;
    if mean_abs_diff > MAX_MEAN_ABS_DIFF {
        return None;
    }
    // Parabolic sub-pixel refinement on each axis independently. Only valid when
    // the integer minimum is strictly inside the search window.
    let mut sub_dx = 0.0_f32;
    let mut sub_dy = 0.0_f32;
    if best_dx > -search_radius && best_dx < search_radius {
        let idx_center =
            ((best_dy + search_radius) as usize) * span + (best_dx + search_radius) as usize;
        let left = sads[idx_center - 1] as f32;
        let center = sads[idx_center] as f32;
        let right = sads[idx_center + 1] as f32;
        let denom = left - 2.0 * center + right;
        if denom > 1e-3 {
            sub_dx = ((left - right) / (2.0 * denom)).clamp(-0.5, 0.5);
        }
    }
    if best_dy > -search_radius && best_dy < search_radius {
        let idx_center =
            ((best_dy + search_radius) as usize) * span + (best_dx + search_radius) as usize;
        let up = sads[idx_center - span] as f32;
        let center = sads[idx_center] as f32;
        let down = sads[idx_center + span] as f32;
        let denom = up - 2.0 * center + down;
        if denom > 1e-3 {
            sub_dy = ((up - down) / (2.0 * denom)).clamp(-0.5, 0.5);
        }
    }
    Some(PatchMatch {
        prev_x: x as f32,
        prev_y: y as f32,
        dx: prior_dx as f32 + best_dx as f32 + sub_dx,
        dy: prior_dy as f32 + best_dy as f32 + sub_dy,
        mean_abs_diff,
    })
}

/// Fit a per-region similarity (translation + uniform scale + rotation) from
/// matches inside each region's footprint, with the *global* dx/dy as a search
/// prior. The output affine maps display-coord points from `previous` to
/// `current`. Regions whose fit doesn't reach `REGION_MIN_INLIERS` come back
/// invalid; callers should fall back to the global motion for those tracks.
fn estimate_per_region(
    previous: &TrackingImage,
    current: &TrackingImage,
    regions: &[Rect],
    global_dx_display: f32,
    global_dy_display: f32,
) -> Vec<LiveRegionMotion> {
    let w = previous.width as i32;
    let h = previous.height as i32;
    let scale_to_full_x = previous.scale_to_full_x;
    let scale_to_full_y = previous.scale_to_full_y;
    let scale_to_track_x = if scale_to_full_x > 0.0 {
        1.0 / scale_to_full_x
    } else {
        1.0
    };
    let scale_to_track_y = if scale_to_full_y > 0.0 {
        1.0 / scale_to_full_y
    } else {
        1.0
    };
    let crop_left = previous.crop.left as f32;
    let crop_top = previous.crop.top as f32;
    let prior_dx_track = (global_dx_display * scale_to_track_x).round() as i32;
    let prior_dy_track = (global_dy_display * scale_to_track_y).round() as i32;

    regions
        .iter()
        .map(|region| {
            estimate_region(
                previous,
                current,
                region,
                w,
                h,
                crop_left,
                crop_top,
                scale_to_track_x,
                scale_to_track_y,
                scale_to_full_x,
                scale_to_full_y,
                prior_dx_track,
                prior_dy_track,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn estimate_region(
    previous: &TrackingImage,
    current: &TrackingImage,
    region: &Rect,
    w: i32,
    h: i32,
    crop_left: f32,
    crop_top: f32,
    scale_to_track_x: f32,
    scale_to_track_y: f32,
    scale_to_full_x: f32,
    scale_to_full_y: f32,
    prior_dx_track: i32,
    prior_dy_track: i32,
) -> LiveRegionMotion {
    let invalid = LiveRegionMotion {
        // Identity affine, so accidental application is a no-op.
        a: 1.0,
        e: 1.0,
        ..LiveRegionMotion::default()
    };
    // Convert region (in display coords) into tracking-image coords (clamp to crop).
    let rx0 = ((region.left as f32 - crop_left) * scale_to_track_x).max(0.0);
    let ry0 = ((region.top as f32 - crop_top) * scale_to_track_y).max(0.0);
    let rx1 = ((region.right as f32 - crop_left) * scale_to_track_x).min(w as f32);
    let ry1 = ((region.bottom as f32 - crop_top) * scale_to_track_y).min(h as f32);
    if rx1 - rx0 < (REGION_GRID_COLS as f32) || ry1 - ry0 < (REGION_GRID_ROWS as f32) {
        return invalid;
    }
    let margin = PATCH_RADIUS + REGION_LOCAL_SEARCH_PX + 1;
    let mut matches = Vec::with_capacity((REGION_GRID_COLS * REGION_GRID_ROWS) as usize);
    for gy in 0..REGION_GRID_ROWS {
        let py = ry0 + (ry1 - ry0) * (gy as f32 + 0.5) / REGION_GRID_ROWS as f32;
        let yi = py.round() as i32;
        if yi - margin < 0 || yi + margin >= h {
            continue;
        }
        for gx in 0..REGION_GRID_COLS {
            let px = rx0 + (rx1 - rx0) * (gx as f32 + 0.5) / REGION_GRID_COLS as f32;
            let xi = px.round() as i32;
            if xi - margin < 0 || xi + margin >= w {
                continue;
            }
            if patch_texture(previous, xi, yi) < MIN_TEXTURE {
                continue;
            }
            if let Some(m) = match_patch(
                previous,
                current,
                xi,
                yi,
                REGION_LOCAL_SEARCH_PX,
                prior_dx_track,
                prior_dy_track,
            ) {
                matches.push(m);
            }
        }
    }
    let total = matches.len();
    if total < REGION_MIN_INLIERS {
        return LiveRegionMotion {
            matches: total as u32,
            ..invalid
        };
    }
    // Inlier rejection against the median displacement: same idea as the global
    // path, but with a tighter radius because we're working in a smaller window.
    let mut dxs: Vec<f32> = matches.iter().map(|m| m.dx).collect();
    let mut dys: Vec<f32> = matches.iter().map(|m| m.dy).collect();
    let med_dx = median(&mut dxs);
    let med_dy = median(&mut dys);
    let inliers: Vec<PatchMatch> = matches
        .iter()
        .filter(|m| {
            let ddx = m.dx - med_dx;
            let ddy = m.dy - med_dy;
            (ddx * ddx + ddy * ddy).sqrt() <= REGION_INLIER_RESIDUAL_PX
        })
        .copied()
        .collect();
    if inliers.len() < REGION_MIN_INLIERS {
        return LiveRegionMotion {
            matches: total as u32,
            inliers: inliers.len() as u32,
            ..invalid
        };
    }
    // Build display-space pairs (p_prev_display, q_curr_display) centred on the
    // region centre for numerical stability.
    let region_cx_display = (region.left as f32 + region.right as f32) * 0.5;
    let region_cy_display = (region.top as f32 + region.bottom as f32) * 0.5;
    let pairs: Vec<(f32, f32, f32, f32)> = inliers
        .iter()
        .map(|m| {
            let prev_disp_x = m.prev_x * scale_to_full_x + crop_left;
            let prev_disp_y = m.prev_y * scale_to_full_y + crop_top;
            let curr_disp_x = (m.prev_x + m.dx) * scale_to_full_x + crop_left;
            let curr_disp_y = (m.prev_y + m.dy) * scale_to_full_y + crop_top;
            (
                prev_disp_x - region_cx_display,
                prev_disp_y - region_cy_display,
                curr_disp_x - region_cx_display,
                curr_disp_y - region_cy_display,
            )
        })
        .collect();

    let Some((aa, bb, tx, ty)) = fit_similarity(&pairs) else {
        return LiveRegionMotion {
            matches: total as u32,
            inliers: inliers.len() as u32,
            ..invalid
        };
    };
    // Build the 2x3 affine in display coords. fit_similarity solves
    //   qx = aa * px - bb * py + tx
    //   qy = bb * px + aa * py + ty
    // so in 2x3-affine form (new_x = a*x + b*y + c, new_y = d*x + e*y + f):
    //   a = aa, b = -bb, d = bb, e = aa
    // The fit is centred on the region centre (rcx, rcy); we expand
    //   q = M*(p - r) + r + t  →  q = M*p + (r - M*r + t)
    // to get the un-centred translation:
    //   c = rcx - (aa*rcx - bb*rcy) + tx
    //   f = rcy - (bb*rcx + aa*rcy) + ty
    let rcx = region_cx_display;
    let rcy = region_cy_display;
    let c = rcx - (aa * rcx - bb * rcy) + tx;
    let f = rcy - (bb * rcx + aa * rcy) + ty;
    LiveRegionMotion {
        valid: true,
        a: aa,
        b: -bb,
        c,
        d: bb,
        e: aa,
        f,
        inliers: inliers.len() as u32,
        matches: total as u32,
    }
}

/// Closed-form similarity fit. Given correspondences `(px, py) -> (qx, qy)`,
/// solves the over-determined system for (A, B, tx, ty) where
///   qx = A * px - B * py + tx
///   qy = B * px + A * py + ty
/// via the standard 4x4 normal equations. Returns `(A, B, tx, ty)` or `None`
/// if the system is too ill-conditioned (insufficient texture spread).
fn fit_similarity(pairs: &[(f32, f32, f32, f32)]) -> Option<(f32, f32, f32, f32)> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f32;
    let mut sx = 0.0_f32;
    let mut sy = 0.0_f32;
    let mut sxx_yy = 0.0_f32;
    let mut sqx = 0.0_f32;
    let mut sqy = 0.0_f32;
    let mut sxq_yq = 0.0_f32;
    let mut sxq_minus_yq = 0.0_f32;
    for &(px, py, qx, qy) in pairs {
        sx += px;
        sy += py;
        sxx_yy += px * px + py * py;
        sqx += qx;
        sqy += qy;
        sxq_yq += px * qx + py * qy;
        sxq_minus_yq += px * qy - py * qx;
    }
    // Closed-form solution: see Umeyama (1991) for the rotation-only case;
    // adapted for centred-only inputs the result reduces to
    //   A = (Σ px*qx + py*qy - (Σ px)(Σ qx)/n - (Σ py)(Σ qy)/n) / D
    //   B = (Σ px*qy - py*qx - (Σ px)(Σ qy)/n + (Σ py)(Σ qx)/n) / D
    //   tx = (Σ qx)/n - A * (Σ px)/n + B * (Σ py)/n
    //   ty = (Σ qy)/n - B * (Σ px)/n - A * (Σ py)/n
    // where D = Σ (px^2 + py^2) - (Σ px)^2/n - (Σ py)^2/n.
    let mean_px = sx / n;
    let mean_py = sy / n;
    let mean_qx = sqx / n;
    let mean_qy = sqy / n;
    let denom = sxx_yy - n * (mean_px * mean_px + mean_py * mean_py);
    if denom <= 1e-3 {
        return None;
    }
    let numer_a = sxq_yq - n * (mean_px * mean_qx + mean_py * mean_qy);
    let numer_b = sxq_minus_yq - n * (mean_px * mean_qy - mean_py * mean_qx);
    let aa = numer_a / denom;
    let bb = numer_b / denom;
    let tx = mean_qx - (aa * mean_px - bb * mean_py);
    let ty = mean_qy - (bb * mean_px + aa * mean_py);
    Some((aa, bb, tx, ty))
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

/// Sample a small grayscale patch from `rgba` over the area covered by an
/// oriented rect in display coordinates. The patch is sampled in the rect's
/// local frame (axis-aligned with the rect), so rotation and uniform scaling
/// of the rect produce the same patch — only content changes affect the
/// output. Used by live-OCR tracks to detect content drift: a track stores
/// a patch at confirmation time and re-samples each frame; a large NCC drop
/// means the rect has slid off its original content and should be retired.
///
/// Sample coordinates are computed in display space, then transformed to
/// sensor space via the same rotation logic the tracker uses, and clamped
/// to the sensor bounds.
pub fn sample_oriented_gray_patch(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
    rotation_degrees: i32,
    rect_cx: f32,
    rect_cy: f32,
    rect_width: f32,
    rect_height: f32,
    rect_angle_radians: f32,
    patch_w: u32,
    patch_h: u32,
) -> Result<Vec<u8>, TranslatorError> {
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
    if patch_w == 0 || patch_h == 0 {
        return Err(TranslatorError::new(
            TranslatorErrorKind::InvalidInput,
            "patch dimensions must be non-zero",
        ));
    }
    let mut out = vec![0u8; (patch_w as usize) * (patch_h as usize)];
    let cos_a = rect_angle_radians.cos();
    let sin_a = rect_angle_radians.sin();
    let hw = rect_width * 0.5;
    let hh = rect_height * 0.5;
    for oy in 0..patch_h {
        let ny = (oy as f32 + 0.5) / patch_h as f32;
        let local_y = (ny * 2.0 - 1.0) * hh;
        for ox in 0..patch_w {
            let nx = (ox as f32 + 0.5) / patch_w as f32;
            let local_x = (nx * 2.0 - 1.0) * hw;
            let display_x = rect_cx + local_x * cos_a - local_y * sin_a;
            let display_y = rect_cy + local_x * sin_a + local_y * cos_a;
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
            out[(oy * patch_w + ox) as usize] = (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Normalized cross-correlation between two equal-length grayscale buffers.
/// Returns a value in [-1, 1] — brightness/contrast invariant. Empty or
/// mismatched-length inputs return 0.0 (treated as "no match" by callers).
pub fn ncc_gray(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    let sum_a: f32 = a.iter().map(|&v| v as f32).sum();
    let sum_b: f32 = b.iter().map(|&v| v as f32).sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let mut num = 0.0_f32;
    let mut den_a = 0.0_f32;
    let mut den_b = 0.0_f32;
    for i in 0..a.len() {
        let da = a[i] as f32 - mean_a;
        let db = b[i] as f32 - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den < 1e-6 {
        0.0
    } else {
        (num / den).clamp(-1.0, 1.0)
    }
}

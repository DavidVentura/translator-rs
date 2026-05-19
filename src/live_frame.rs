//! Live OCR frame buffer: holds the raw RGBA sensor frame plus a
//! lazily-built cropped + rotated derived form (RGB + grayscale) so
//! detection and recognition can both operate on the same in-memory
//! image without re-shuffling bytes across the FFI boundary per call.
//!
//! Per-frame planar-tracker step needs only `gray` and uses
//! [`OrientedImage::build`], which produces it via a single fused
//! crop+rotate+RGBA→luma pass (~5–8 ms at 1.2 MP vs the legacy
//! ~33 ms scalar four-pass chain).
//!
//! Detect / recognize paths additionally need `rgb` and `rgb_det`
//! and call [`OrientedImage::build_with_rgb`] which adds the scalar
//! rotate-RGB + det-downscale chain on top.
//!
//! This is the Rust-side primitive backing the `FrameHandle` exposed
//! via uniffi.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::ocr::Rect;
use crate::ppocr::rgba_to_dynamic;

/// A frame's cropped, rotated, and detection-ready derivatives.
///
/// `gray` is always populated. `rgb` and `rgb_det` are `Some` iff this
/// was built via [`Self::build_with_rgb`] — the per-frame
/// planar-tracker step uses the faster gray-only path and leaves them
/// `None`.
///
/// `det_to_full_scale` is meaningful iff `rgb_det.is_some()`.
pub struct OrientedImage {
    pub gray: GrayImage,
    pub display_crop: Rect,
    pub rgb: Option<DynamicImage>,
    pub rgb_det: Option<DynamicImage>,
    pub det_to_full_scale: f32,
}

impl OrientedImage {
    /// Fast path: build only `gray` via a single fused
    /// crop+rotate+RGBA→luma pass. Used by the per-frame planar
    /// tracker step which never reads `rgb` / `rgb_det`. Skips ~20–25
    /// ms of scalar rotate-RGB + grayscale-from-RGB + resize_exact at
    /// 1.2 MP compared to [`Self::build_with_rgb`].
    pub fn build(
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
        _det_max_pixels: u32,
    ) -> Result<Self, TranslatorError> {
        validate_rgba_len(rgba, sensor_width, sensor_height)?;
        let gray = build_gray_fused(
            rgba,
            sensor_width,
            sensor_height,
            rotation_degrees,
            display_crop,
        )?;
        Ok(OrientedImage {
            gray,
            display_crop,
            rgb: None,
            rgb_det: None,
            det_to_full_scale: 1.0,
        })
    }

    /// Eager build for callers (detect / recognize / matting) that
    /// need the colour image as well. Uses the same fused gray pass
    /// plus a scalar rotate-RGB chain for `rgb` and a bilinear
    /// downscale for `rgb_det`.
    pub fn build_with_rgb(
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
        det_max_pixels: u32,
    ) -> Result<Self, TranslatorError> {
        validate_rgba_len(rgba, sensor_width, sensor_height)?;
        let gray = build_gray_fused(
            rgba,
            sensor_width,
            sensor_height,
            rotation_degrees,
            display_crop,
        )?;
        let rgb_full = build_rgb_full(
            rgba,
            sensor_width,
            sensor_height,
            rotation_degrees,
            display_crop,
        )?;
        let display_w = rgb_full.width();
        let display_h = rgb_full.height();
        let full_pixels = (display_w as u64) * (display_h as u64);
        let (det_scale, rgb_det) = if full_pixels > det_max_pixels as u64 {
            let scale = (det_max_pixels as f64 / full_pixels as f64).sqrt() as f32;
            let new_w = ((display_w as f32) * scale).max(1.0) as u32;
            let new_h = ((display_h as f32) * scale).max(1.0) as u32;
            let det = rgb_full.resize_exact(new_w, new_h, FilterType::Triangle);
            (scale, det)
        } else {
            (1.0_f32, rgb_full.clone())
        };
        let det_to_full_scale = if det_scale > 0.0 {
            1.0 / det_scale
        } else {
            1.0
        };
        Ok(OrientedImage {
            gray,
            display_crop,
            rgb: Some(rgb_full),
            rgb_det: Some(rgb_det),
            det_to_full_scale,
        })
    }

    /// True iff `rgb` / `rgb_det` are populated. Used by the cache
    /// validator to detect "we built gray-only last frame but the
    /// caller this frame needs rgb — rebuild."
    pub fn has_rgb(&self) -> bool {
        self.rgb.is_some()
    }
}

fn validate_rgba_len(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
) -> Result<(), TranslatorError> {
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
    Ok(())
}

/// Single-pass fused crop+rotate+RGBA→luma. Walks the destination
/// gray buffer in row-major display orientation and derives the
/// source RGBA byte address per output pixel from the rotation.
/// Integer Rec.601 luma `(77·R + 150·G + 29·B + 128) >> 8`.
///
/// Replaces the legacy four-pass chain (crop RGBA→RGB, rotate90,
/// grayscale, resize_exact for the gray case): ~33 ms → ~5–8 ms at
/// 1.2 MP on a phone CPU, scalar. NEON acceleration is possible
/// later but the source read traffic dominates so a SIMD luma op
/// alone won't help much.
fn build_gray_fused(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
    rotation_degrees: i32,
    display_crop: Rect,
) -> Result<GrayImage, TranslatorError> {
    let r = ((rotation_degrees % 360) + 360) % 360;
    if !matches!(r, 0 | 90 | 180 | 270) {
        return Err(TranslatorError::new(
            TranslatorErrorKind::InvalidInput,
            format!("unsupported rotation_degrees: {}", rotation_degrees),
        ));
    }
    let (display_w, display_h) = if r == 90 || r == 270 {
        (sensor_height, sensor_width)
    } else {
        (sensor_width, sensor_height)
    };
    let right = display_crop.right.min(display_w);
    let bottom = display_crop.bottom.min(display_h);
    if right <= display_crop.left || bottom <= display_crop.top {
        return Err(TranslatorError::new(
            TranslatorErrorKind::InvalidInput,
            "crop region is empty after clamp",
        ));
    }
    let crop_w = right - display_crop.left;
    let crop_h = bottom - display_crop.top;
    let total = (crop_w as usize) * (crop_h as usize);
    let mut gray = vec![0u8; total];

    let sw = sensor_width as i32;
    let sh = sensor_height as i32;
    let crop_left = display_crop.left as i32;
    let crop_top = display_crop.top as i32;
    let crop_w_i = crop_w as i32;
    let stride = (sensor_width as usize) * 4;

    // For each rotation, compute the (sx0, sy0) source coord for
    // dst (dx=0, dy=current row) and the (sdx_x, sdx_y) step taken
    // per inner-loop column. Pulls the rotation branch out of the
    // per-pixel hot path.
    for dy in 0..crop_h as i32 {
        let display_y = crop_top + dy;
        let dst_row_off = (dy as usize) * (crop_w as usize);
        let (sx0, sy0, sdx_x, sdx_y) = match r {
            0 => (crop_left, display_y, 1i32, 0i32),
            90 => (display_y, sh - 1 - crop_left, 0i32, -1i32),
            180 => (sw - 1 - crop_left, sh - 1 - display_y, -1i32, 0i32),
            _ /* 270 */ => (sw - 1 - display_y, crop_left, 0i32, 1i32),
        };
        let mut sx = sx0;
        let mut sy = sy0;
        for dx in 0..crop_w_i {
            let src_idx = (sy as usize) * stride + (sx as usize) * 4;
            let rr = rgba[src_idx] as u32;
            let gg = rgba[src_idx + 1] as u32;
            let bb = rgba[src_idx + 2] as u32;
            // sRGB luma per BT.709 (matches `image::imageops::grayscale`'s
            // coefficients exactly so the tracker's downstream FAST+BRIEF
            // sees bit-identical input to the old path). 16-bit fixed
            // point: `0.2126 ≈ 13933 / 65536`, etc.
            let luma = ((13933 * rr + 46871 * gg + 4732 * bb) >> 16) as u8;
            gray[dst_row_off + dx as usize] = luma;
            sx += sdx_x;
            sy += sdx_y;
        }
    }

    GrayImage::from_raw(crop_w, crop_h, gray).ok_or_else(|| {
        TranslatorError::new(TranslatorErrorKind::Internal, "gray buffer size mismatch")
    })
}

fn build_rgb_full(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
    rotation_degrees: i32,
    display_crop: Rect,
) -> Result<DynamicImage, TranslatorError> {
    let sensor_crop =
        display_crop_to_sensor(display_crop, sensor_width, sensor_height, rotation_degrees)?;
    let crop_w = sensor_crop.right - sensor_crop.left;
    let crop_h = sensor_crop.bottom - sensor_crop.top;
    let mut rgb_bytes = Vec::with_capacity((crop_w * crop_h * 3) as usize);
    for y in sensor_crop.top..sensor_crop.bottom {
        let row_start = (y * sensor_width * 4) as usize;
        for x in sensor_crop.left..sensor_crop.right {
            let px = row_start + (x as usize * 4);
            rgb_bytes.push(rgba[px]);
            rgb_bytes.push(rgba[px + 1]);
            rgb_bytes.push(rgba[px + 2]);
        }
    }
    let sensor_rgb = RgbImage::from_raw(crop_w, crop_h, rgb_bytes).ok_or_else(|| {
        TranslatorError::new(TranslatorErrorKind::Internal, "rgb buffer size mismatch")
    })?;
    let rotated = rotate_rgb(sensor_rgb, rotation_degrees);
    Ok(DynamicImage::ImageRgb8(rotated))
}

/// Apply the inverse of a sensor→display rotation to a display-orient rect to
/// recover the corresponding sensor-orient rect. Both rects are in pixel coords.
fn display_crop_to_sensor(
    crop: Rect,
    sensor_w: u32,
    sensor_h: u32,
    rotation_degrees: i32,
) -> Result<Rect, TranslatorError> {
    let r = ((rotation_degrees % 360) + 360) % 360;
    let out = match r {
        0 => Rect {
            left: crop.left.min(sensor_w),
            top: crop.top.min(sensor_h),
            right: crop.right.min(sensor_w),
            bottom: crop.bottom.min(sensor_h),
        },
        90 => Rect {
            left: crop.top.min(sensor_w),
            top: sensor_h.saturating_sub(crop.right),
            right: crop.bottom.min(sensor_w),
            bottom: sensor_h.saturating_sub(crop.left),
        },
        180 => Rect {
            left: sensor_w.saturating_sub(crop.right),
            top: sensor_h.saturating_sub(crop.bottom),
            right: sensor_w.saturating_sub(crop.left),
            bottom: sensor_h.saturating_sub(crop.top),
        },
        270 => Rect {
            left: sensor_w.saturating_sub(crop.bottom),
            top: crop.left.min(sensor_h),
            right: sensor_w.saturating_sub(crop.top),
            bottom: crop.right.min(sensor_h),
        },
        _ => {
            return Err(TranslatorError::new(
                TranslatorErrorKind::InvalidInput,
                format!("unsupported rotation_degrees: {}", rotation_degrees),
            ));
        }
    };
    if out.right <= out.left || out.bottom <= out.top {
        return Err(TranslatorError::new(
            TranslatorErrorKind::InvalidInput,
            "crop region is empty after sensor mapping",
        ));
    }
    Ok(out)
}

fn rotate_rgb(image: RgbImage, rotation_degrees: i32) -> RgbImage {
    use image::imageops;
    let r = ((rotation_degrees % 360) + 360) % 360;
    match r {
        90 => imageops::rotate90(&image),
        180 => imageops::rotate180(&image),
        270 => imageops::rotate270(&image),
        _ => image,
    }
}

#[allow(dead_code)] // exposed for tests / future direct callers
pub fn build_oriented_image(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
    rotation_degrees: i32,
    display_crop: Rect,
    det_max_pixels: u32,
) -> Result<OrientedImage, TranslatorError> {
    OrientedImage::build_with_rgb(
        rgba,
        sensor_width,
        sensor_height,
        rotation_degrees,
        display_crop,
        det_max_pixels,
    )
}

/// Forwards to `rgba_to_dynamic` so external callers (tests, bindings) can build
/// a DynamicImage without going through the live-frame pipeline.
pub fn rgba_bytes_to_dynamic(rgba: &[u8], width: u32, height: u32) -> DynamicImage {
    rgba_to_dynamic(rgba, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b, 255]);
        }
        v
    }

    #[test]
    fn gray_solid_color_rotation_zero() {
        let rgba = make_solid_rgba(8, 4, 128, 64, 200);
        let crop = Rect {
            left: 0,
            top: 0,
            right: 8,
            bottom: 4,
        };
        let oi = OrientedImage::build(&rgba, 8, 4, 0, crop, 1_000_000).unwrap();
        assert_eq!(oi.gray.dimensions(), (8, 4));
        assert!(oi.rgb.is_none());
        // BT.709 luma at 16-bit fixed point.
        let expected = ((13933u32 * 128 + 46871 * 64 + 4732 * 200) >> 16) as u8;
        for &b in oi.gray.as_raw() {
            assert_eq!(b, expected);
        }
    }

    #[test]
    fn gray_dims_swap_under_90_rotation() {
        let rgba = make_solid_rgba(8, 4, 0, 0, 0);
        let crop = Rect {
            left: 0,
            top: 0,
            right: 4,
            bottom: 8,
        };
        let oi = OrientedImage::build(&rgba, 8, 4, 90, crop, 1_000_000).unwrap();
        assert_eq!(oi.gray.dimensions(), (4, 8));
    }

    /// The fused fast path produces the same bytes (±1 LSB rounding)
    /// as the legacy rotate-then-grayscale chain, for every rotation.
    /// Catches sign/offset mistakes in the source-pointer arithmetic.
    #[test]
    fn fused_matches_legacy_for_all_rotations() {
        let sensor_w = 6u32;
        let sensor_h = 4u32;
        let mut rgba = Vec::with_capacity((sensor_w * sensor_h * 4) as usize);
        for y in 0..sensor_h {
            for x in 0..sensor_w {
                rgba.extend_from_slice(&[
                    (x * 30) as u8,
                    (y * 50) as u8,
                    ((x + y) * 20) as u8,
                    255,
                ]);
            }
        }
        for rot in [0, 90, 180, 270] {
            let (dw, dh) = if rot == 90 || rot == 270 {
                (sensor_h, sensor_w)
            } else {
                (sensor_w, sensor_h)
            };
            let crop = Rect {
                left: 0,
                top: 0,
                right: dw,
                bottom: dh,
            };
            let legacy_rgb = build_rgb_full(&rgba, sensor_w, sensor_h, rot, crop).unwrap();
            let legacy_gray = image::imageops::grayscale(&legacy_rgb.to_rgb8());

            let oi = OrientedImage::build(&rgba, sensor_w, sensor_h, rot, crop, 1_000_000).unwrap();
            for (got, want) in oi.gray.as_raw().iter().zip(legacy_gray.as_raw().iter()) {
                let diff = (*got as i32 - *want as i32).abs();
                assert!(
                    diff <= 1,
                    "rotation {}: luma mismatch {} vs {}",
                    rot,
                    got,
                    want,
                );
            }
        }
    }

    #[test]
    fn with_rgb_populates_rgb_and_det() {
        let rgba = make_solid_rgba(4, 4, 200, 100, 50);
        let crop = Rect {
            left: 0,
            top: 0,
            right: 4,
            bottom: 4,
        };
        let oi = OrientedImage::build_with_rgb(&rgba, 4, 4, 0, crop, 1_000_000).unwrap();
        assert!(oi.rgb.is_some());
        assert!(oi.rgb_det.is_some());
    }
}

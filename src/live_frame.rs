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
    /// Cache key from the caller (visible region in display coords).
    /// Kept verbatim so the eq-check that drives cache reuse still
    /// matches what the Kotlin side passes.
    pub display_crop: Rect,
    /// Same region projected into sensor coords. Source of truth for
    /// "where in the full sensor frame these derived buffers live" —
    /// `gray`/`rgb`/`rgb_det` are all sensor-orient, sized to this
    /// rect. Callers translate PPOCR boxes into full-sensor coords by
    /// adding `sensor_crop.left/top`.
    pub sensor_crop: Rect,
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
    ///
    /// When the fused gray exceeds `det_max_pixels`, it's downsampled
    /// with a triangle filter and `det_to_full_scale` is set to the
    /// inverse of the applied scale (i.e. the multiplier that maps a
    /// small-coord point back to full-display coords). Per-frame
    /// tracker matching cost is linear in pixel count, and the planar
    /// tracker doesn't need full-res to find robust correspondences,
    /// so capping here is a direct ~2× perf win.
    pub fn build(
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
        det_max_pixels: u32,
    ) -> Result<Self, TranslatorError> {
        validate_rgba_len(rgba, sensor_width, sensor_height)?;
        let sensor_crop =
            display_crop_to_sensor(display_crop, sensor_width, sensor_height, rotation_degrees)?;
        let crop_w = sensor_crop.right - sensor_crop.left;
        let crop_h = sensor_crop.bottom - sensor_crop.top;
        let full_pixels = (crop_w as u64) * (crop_h as u64);
        let (gray, det_to_full_scale) = if full_pixels > det_max_pixels as u64 {
            let scale = (det_max_pixels as f64 / full_pixels as f64).sqrt() as f32;
            let target_w = ((crop_w as f32) * scale).max(1.0) as u32;
            let target_h = ((crop_h as f32) * scale).max(1.0) as u32;
            let gray =
                build_gray_fused_downsampled(rgba, sensor_width, &sensor_crop, target_w, target_h)?;
            (gray, 1.0 / scale)
        } else {
            let gray = build_gray_fused(
                rgba,
                sensor_width,
                sensor_height,
                rotation_degrees,
                display_crop,
            )?;
            (gray, 1.0)
        };
        Ok(OrientedImage {
            gray,
            display_crop,
            sensor_crop,
            rgb: None,
            rgb_det: None,
            det_to_full_scale,
        })
    }

    /// Eager build for callers (detect / recognize / matting) that
    /// need the colour image as well. Uses the same fused gray pass
    /// plus a sensor-orient RGB build (no rotation) and a bilinear
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
        let sensor_crop =
            display_crop_to_sensor(display_crop, sensor_width, sensor_height, rotation_degrees)?;
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
            sensor_crop,
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

/// Single-pass crop + RGBA→luma in **sensor orientation**. Walks the
/// sensor-orient crop sequentially row-by-row — no rotation, purely
/// stride-1 source reads, prefetcher-friendly. Integer BT.709 luma
/// `(13933·R + 46871·G + 4732·B) >> 16` (matches
/// `image::imageops::grayscale` to ±1 LSB so downstream FAST+BRIEF
/// sees bit-identical input to the old path).
///
/// `display_crop` is in display-orient coords; we convert to the
/// equivalent sensor-orient rect via `display_crop_to_sensor` so the
/// caller's existing crop semantics work without rewiring (and crop
/// is currently full-frame anyway). The output gray buffer's
/// dimensions are the *sensor-orient* crop size — i.e. width/height
/// swap relative to the legacy display-orient gray under R90/R270.
/// The tracker now operates on this sensor-orient gray; the
/// per-frame rotation pass is gone (saves ~3 ms at 1.2 MP).
fn build_gray_fused(
    rgba: &[u8],
    sensor_width: u32,
    sensor_height: u32,
    rotation_degrees: i32,
    display_crop: Rect,
) -> Result<GrayImage, TranslatorError> {
    let sensor_crop =
        display_crop_to_sensor(display_crop, sensor_width, sensor_height, rotation_degrees)?;
    let crop_w = sensor_crop.right - sensor_crop.left;
    let crop_h = sensor_crop.bottom - sensor_crop.top;
    let total = (crop_w as usize) * (crop_h as usize);
    let mut gray = vec![0u8; total];

    let stride = (sensor_width as usize) * 4;
    let crop_left = sensor_crop.left as usize;
    let crop_w_usize = crop_w as usize;

    for sy in sensor_crop.top..sensor_crop.bottom {
        let src_row = (sy as usize) * stride;
        let dst_row = ((sy - sensor_crop.top) as usize) * crop_w_usize;
        for dx in 0..crop_w_usize {
            let p = src_row + (crop_left + dx) * 4;
            let rr = rgba[p] as u32;
            let gg = rgba[p + 1] as u32;
            let bb = rgba[p + 2] as u32;
            let luma = ((13933 * rr + 46871 * gg + 4732 * bb) >> 16) as u8;
            gray[dst_row + dx] = luma;
        }
    }

    GrayImage::from_raw(crop_w, crop_h, gray).ok_or_else(|| {
        TranslatorError::new(TranslatorErrorKind::Internal, "gray buffer size mismatch")
    })
}

/// Single-pass crop + nearest-neighbor-downsample + RGBA→luma in
/// **sensor orientation**. One source RGBA read per output pixel — the
/// total byte traffic is smaller than the full-res fused-luma path
/// because we're producing fewer pixels, so this is a net per-frame win
/// (not just a wash with the engine speedup). Nearest is fine for the
/// tracker because BRIEF descriptors are binary pair-tests over a
/// 31-pixel patch, robust to the mild aliasing nearest introduces at
/// the ~1.33× ratio we use here.
fn build_gray_fused_downsampled(
    rgba: &[u8],
    sensor_width: u32,
    sensor_crop: &Rect,
    target_w: u32,
    target_h: u32,
) -> Result<GrayImage, TranslatorError> {
    let crop_w = sensor_crop.right - sensor_crop.left;
    let crop_h = sensor_crop.bottom - sensor_crop.top;
    let stride = (sensor_width as usize) * 4;
    // Precompute source-x for every target column once (16-bit fixed
    // point isn't worth the trouble at this size, but the precomputed
    // table avoids a multiply+divide per output pixel and is cache-hot).
    let mut sx_table = vec![0usize; target_w as usize];
    for tx in 0..target_w {
        let sx_crop = ((tx as u64) * (crop_w as u64) / (target_w as u64)) as u32;
        sx_table[tx as usize] = (sensor_crop.left + sx_crop) as usize;
    }
    let total = (target_w as usize) * (target_h as usize);
    let mut gray = vec![0u8; total];
    for ty in 0..target_h {
        let sy_crop = ((ty as u64) * (crop_h as u64) / (target_h as u64)) as u32;
        let sy = (sensor_crop.top + sy_crop) as usize;
        let src_row = sy * stride;
        let dst_row = (ty as usize) * (target_w as usize);
        for tx in 0..target_w as usize {
            let p = src_row + sx_table[tx] * 4;
            let rr = rgba[p] as u32;
            let gg = rgba[p + 1] as u32;
            let bb = rgba[p + 2] as u32;
            // Same BT.709 coefficients as `build_gray_fused` so anchor
            // and per-frame descriptors see consistent luma.
            let luma = ((13933 * rr + 46871 * gg + 4732 * bb) >> 16) as u8;
            gray[dst_row + tx] = luma;
        }
    }
    GrayImage::from_raw(target_w, target_h, gray).ok_or_else(|| {
        TranslatorError::new(
            TranslatorErrorKind::Internal,
            "downsampled gray buffer size mismatch",
        )
    })
}

/// Crops the sensor RGBA to the sensor-space rect equivalent to
/// `display_crop` and returns RGB **in sensor orientation** (no rotation).
/// The Kotlin SurfaceView rotates the final composite for display at
/// scanout; the OCR pipeline operates entirely in sensor frame.
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
    Ok(DynamicImage::ImageRgb8(sensor_rgb))
}

/// 3×3 homography mapping display-orient pixel coords to sensor-orient
/// pixel coords under the given camera rotation. Composed into the
/// `h_view_to_surface` matrix at the PPOCR boundary so detected text
/// boxes (in display coords, since PPOCR runs on display-orient RGB)
/// project correctly into the tracker's sensor-orient surface frame
/// via one matrix multiply.
///
/// Row-major. `rotation_degrees` is the CameraX
/// `ImageInfo.rotationDegrees` (the sensor→display rotation).
pub fn display_to_sensor_homography(
    sensor_w: u32,
    sensor_h: u32,
    rotation_degrees: i32,
) -> [f32; 9] {
    let r = ((rotation_degrees % 360) + 360) % 360;
    let sw = sensor_w as f32;
    let sh = sensor_h as f32;
    match r {
        0 => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        // display(dx, dy) → sensor(dy, sh - dx)
        90 => [0.0, 1.0, 0.0, -1.0, 0.0, sh, 0.0, 0.0, 1.0],
        // display(dx, dy) → sensor(sw - dx, sh - dy)
        180 => [-1.0, 0.0, sw, 0.0, -1.0, sh, 0.0, 0.0, 1.0],
        // display(dx, dy) → sensor(sw - dy, dx)
        270 => [0.0, -1.0, sw, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        _ => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
}

/// Apply the inverse of a sensor→display rotation to a display-orient rect to
/// recover the corresponding sensor-orient rect. Both rects are in pixel coords.
///
/// Public so the bindings layer can use it to convert PPOCR-detected
/// boxes (display-orient, since PPOCR runs on display-orient RGB) into
/// sensor-orient regions for the tracker's anchor builder (which works
/// on sensor-orient gray).
pub fn display_crop_to_sensor(
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

/// Build a DynamicImage without going through the live-frame pipeline.
pub fn rgba_bytes_to_dynamic(rgba: &[u8], width: u32, height: u32) -> DynamicImage {
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

    /// Gray is in **sensor orientation** now (no per-frame rotation).
    /// For an (8×4) sensor under R90, a display crop of (4×8) covers
    /// the whole sensor → gray comes out at sensor dims (8×4), not
    /// the display dims (4×8) the legacy code produced.
    #[test]
    fn gray_is_sensor_orient_under_90_rotation() {
        let rgba = make_solid_rgba(8, 4, 0, 0, 0);
        let crop = Rect {
            left: 0,
            top: 0,
            right: 4,
            bottom: 8,
        };
        let oi = OrientedImage::build(&rgba, 8, 4, 90, crop, 1_000_000).unwrap();
        assert_eq!(oi.gray.dimensions(), (8, 4));
    }

    /// Gray luma matches `image::imageops::grayscale` of the
    /// sensor-orient RGB (no rotation) to ±1 LSB. Catches
    /// coefficient mistakes in the BT.709 fixed-point math.
    #[test]
    fn fused_gray_matches_legacy_no_rotation() {
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
            // Display crop covering the whole display-orient frame
            // (which is sensor swapped under R90/R270).
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
            let oi = OrientedImage::build(&rgba, sensor_w, sensor_h, rot, crop, 1_000_000).unwrap();
            // Expected: sensor-orient gray of the whole RGBA, no rotation.
            let mut expected_rgb = Vec::with_capacity((sensor_w * sensor_h * 3) as usize);
            for chunk in rgba.chunks_exact(4) {
                expected_rgb.push(chunk[0]);
                expected_rgb.push(chunk[1]);
                expected_rgb.push(chunk[2]);
            }
            let expected_gray = image::imageops::grayscale(
                &RgbImage::from_raw(sensor_w, sensor_h, expected_rgb).unwrap(),
            );
            assert_eq!(oi.gray.dimensions(), expected_gray.dimensions());
            for (got, want) in oi.gray.as_raw().iter().zip(expected_gray.as_raw().iter()) {
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

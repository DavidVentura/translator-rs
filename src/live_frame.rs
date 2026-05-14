//! Live OCR frame buffer: holds the raw RGBA sensor frame plus a lazily-built
//! cropped + rotated derived form (RGB + grayscale) so detection and recognition
//! can both operate on the same in-memory image without re-shuffling bytes across
//! the FFI boundary per call.
//!
//! This is the Rust-side primitive backing the `FrameHandle` exposed via uniffi.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};

use crate::api::{TranslatorError, TranslatorErrorKind};
use crate::ocr::Rect;
use crate::ppocr::rgba_to_dynamic;

/// A frame's cropped, rotated, and detection-ready derivatives. Built once per
/// crop region per frame; recognised again from the cached `rgb`/`gray` whenever
/// the box list changes.
pub struct OrientedImage {
    /// Full-resolution crop in display orientation.
    pub rgb: DynamicImage,
    /// Grayscale of `rgb`. Pre-computed so per-box dewarp doesn't recompute it.
    pub gray: GrayImage,
    /// Detection-sized downscaled image (≤ `det_max_pixels`).
    pub rgb_det: DynamicImage,
    /// Multiply det-image coords by this to get coords in `rgb`.
    pub det_to_full_scale: f32,
    /// What display-orient crop this was built for (for cache validation).
    pub display_crop: Rect,
}

impl OrientedImage {
    /// Slice the requested display-coord crop region from the sensor-oriented
    /// `rgba` buffer, rotate to display orientation, and build the derived
    /// detection / grayscale images. `rotation_degrees` is the sensor → display
    /// rotation reported by the camera framework (0/90/180/270).
    pub fn build(
        rgba: &[u8],
        sensor_width: u32,
        sensor_height: u32,
        rotation_degrees: i32,
        display_crop: Rect,
        det_max_pixels: u32,
    ) -> Result<Self, TranslatorError> {
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

        // Convert the display-orient crop rect into a sensor-orient crop rect by
        // applying the inverse of the sensor→display rotation.
        let sensor_crop =
            display_crop_to_sensor(display_crop, sensor_width, sensor_height, rotation_degrees)?;

        // Slice the rgba buffer into a sensor-orient RGB image of just the crop region.
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

        // Rotate the small crop to display orientation.
        let rotated = rotate_rgb(sensor_rgb, rotation_degrees);
        let display_w = rotated.width();
        let display_h = rotated.height();

        // Pre-build grayscale view for dewarp. Compute *before* moving `rotated`
        // into the DynamicImage so we don't need a clone.
        let gray = image::imageops::grayscale(&rotated);
        let rgb_full = DynamicImage::ImageRgb8(rotated);

        // Det-target downscale (filter Triangle = bilinear). Sized to hit ≤ max pixels.
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
            rgb: rgb_full,
            gray,
            rgb_det,
            det_to_full_scale,
            display_crop,
        })
    }
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
    OrientedImage::build(
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

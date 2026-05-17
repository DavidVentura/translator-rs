//! Composite the camera frame and one or more overlay items into a
//! display-orient RGBA buffer that the Kotlin side hands straight to a
//! [`SurfaceView`]. Replaces the previous "preview surface + separate
//! overlay view" architecture, where the two streams updated at
//! different rates and the overlay could visibly drift from the camera
//! pixels under motion.
//!
//! Phase 1 callers pass a single item — a pre-rasterized text-overlay
//! bitmap covering the union of all overlay quads of the currently
//! locked anchor. The API takes a `Vec<OverlayItem>` so phase 2's
//! sliding-window content map can ship per-block items in surface
//! coordinates without an API churn.
//!
//! All "surface coords" in this module mean the planar tracker's
//! canonical-frame coords (== display-orient crop coords as long as
//! CENTER_CROP_FRACTION_PLANAR is 1.0). `h_surface_to_viewport` is the
//! row-major 3x3 mapping surface points to current-frame pixel coords.

use crate::homography::{invert, mat3_mul, project};

/// One overlay payload to be drawn onto the camera frame.
///
/// `bitmap_rgba` is a pre-rasterized RGBA8888 region. Its pixel (0, 0)
/// corresponds to surface coordinate
/// `(bitmap_origin_surface_x, bitmap_origin_surface_y)`. Its width and
/// height are in surface pixels — i.e. the bitmap is conceptually
/// "drawn on the surface" at those bounds, and projection through
/// `h_surface_to_viewport` decides where it lands in the camera frame.
pub struct OverlayItem<'a> {
    pub bitmap_rgba: &'a [u8],
    pub bitmap_width: u32,
    pub bitmap_height: u32,
    pub bitmap_origin_surface_x: f32,
    pub bitmap_origin_surface_y: f32,
}

#[derive(Debug)]
pub enum CompositeError {
    DstBufferSize,
    SrcBufferSize,
    UnsupportedRotation,
    DimensionMismatch,
}

/// Build the per-frame display image: rotate the sensor RGBA into the
/// display orientation, then warp+blend each overlay item on top.
///
/// `dst_rgba` must be sized `display_w * display_h * 4`. On success
/// it's fully overwritten with the composited result and can be blitted
/// to a `SurfaceView` directly.
pub fn composite_frame_into(
    dst_rgba: &mut [u8],
    display_w: u32,
    display_h: u32,
    camera_rgba: &[u8],
    sensor_w: u32,
    sensor_h: u32,
    rotation_degrees: i32,
    h_surface_to_viewport: &[f32; 9],
    items: &[OverlayItem<'_>],
) -> Result<(), CompositeError> {
    let dst_bytes = (display_w as usize)
        .checked_mul(display_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::DstBufferSize)?;
    if dst_rgba.len() != dst_bytes {
        return Err(CompositeError::DstBufferSize);
    }
    let src_bytes = (sensor_w as usize)
        .checked_mul(sensor_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::SrcBufferSize)?;
    if camera_rgba.len() != src_bytes {
        return Err(CompositeError::SrcBufferSize);
    }
    rotate_camera_into_display(
        dst_rgba,
        display_w,
        display_h,
        camera_rgba,
        sensor_w,
        sensor_h,
        rotation_degrees,
    )?;
    for item in items {
        warp_item_onto_display(dst_rgba, display_w, display_h, item, h_surface_to_viewport);
    }
    Ok(())
}

/// Sensor-to-display rotation using the CameraX convention for
/// `imageInfo.rotationDegrees`: the rotation that, when applied to the
/// sensor image, yields the natural display orientation.
fn rotate_camera_into_display(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    rotation_degrees: i32,
) -> Result<(), CompositeError> {
    let r = ((rotation_degrees % 360) + 360) % 360;
    match r {
        0 => {
            if (dst_w, dst_h) != (src_w, src_h) {
                return Err(CompositeError::DimensionMismatch);
            }
            dst.copy_from_slice(src);
        }
        90 => {
            if (dst_w, dst_h) != (src_h, src_w) {
                return Err(CompositeError::DimensionMismatch);
            }
            // Rotate 90° clockwise: dst[y][x] = src[src_h - 1 - x][y].
            for y in 0..dst_h {
                for x in 0..dst_w {
                    let src_x = y;
                    let src_y = src_h - 1 - x;
                    let s = ((src_y * src_w + src_x) * 4) as usize;
                    let d = ((y * dst_w + x) * 4) as usize;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
        180 => {
            if (dst_w, dst_h) != (src_w, src_h) {
                return Err(CompositeError::DimensionMismatch);
            }
            for y in 0..dst_h {
                for x in 0..dst_w {
                    let src_x = src_w - 1 - x;
                    let src_y = src_h - 1 - y;
                    let s = ((src_y * src_w + src_x) * 4) as usize;
                    let d = ((y * dst_w + x) * 4) as usize;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
        270 => {
            if (dst_w, dst_h) != (src_h, src_w) {
                return Err(CompositeError::DimensionMismatch);
            }
            // Rotate 270° clockwise (= 90° CCW): dst[y][x] = src[x][src_w - 1 - y].
            for y in 0..dst_h {
                for x in 0..dst_w {
                    let src_x = src_w - 1 - y;
                    let src_y = x;
                    let s = ((src_y * src_w + src_x) * 4) as usize;
                    let d = ((y * dst_w + x) * 4) as usize;
                    dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
                }
            }
        }
        _ => return Err(CompositeError::UnsupportedRotation),
    }
    Ok(())
}

/// Perspective-warp an item's bitmap onto the display buffer with
/// bilinear sampling and source-over alpha compositing.
///
/// We compose `bitmap_to_viewport = H_surface_to_viewport ·
/// translate(origin)` so a single homography maps bitmap pixel coords
/// directly to viewport coords. Then we invert it for backward
/// sampling: iterate the projected AABB in the viewport, for each pixel
/// project back to the bitmap, sample bilinearly, blend.
fn warp_item_onto_display(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    item: &OverlayItem<'_>,
    h_surface_to_viewport: &[f32; 9],
) {
    let src_w = item.bitmap_width;
    let src_h = item.bitmap_height;
    let src = item.bitmap_rgba;
    if src.is_empty() || src_w == 0 || src_h == 0 {
        return;
    }
    let expected_src = (src_w as usize) * (src_h as usize) * 4;
    if src.len() != expected_src {
        return;
    }

    let ox = item.bitmap_origin_surface_x;
    let oy = item.bitmap_origin_surface_y;
    let bitmap_to_surface = [1.0, 0.0, ox, 0.0, 1.0, oy, 0.0, 0.0, 1.0];
    let bitmap_to_viewport = mat3_mul(h_surface_to_viewport, &bitmap_to_surface);
    let viewport_to_bitmap = match invert(&bitmap_to_viewport) {
        Some(v) => v,
        None => return,
    };

    // Forward-project the bitmap's pixel corners to find the AABB of the
    // affected region in the viewport. Clip to the buffer.
    let src_corners = [
        (0.0_f32, 0.0_f32),
        (src_w as f32, 0.0),
        (src_w as f32, src_h as f32),
        (0.0, src_h as f32),
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &(sx, sy) in &src_corners {
        let (px, py) = match project(&bitmap_to_viewport, sx, sy) {
            Some(p) => p,
            None => return,
        };
        if !px.is_finite() || !py.is_finite() {
            return;
        }
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px);
        max_y = max_y.max(py);
    }
    let x0 = (min_x.floor().max(0.0) as u32).min(dst_w);
    let y0 = (min_y.floor().max(0.0) as u32).min(dst_h);
    let x1 = ((max_x.ceil()).max(0.0) as u32).min(dst_w);
    let y1 = ((max_y.ceil()).max(0.0) as u32).min(dst_h);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    let src_max_x = (src_w - 1) as f32;
    let src_max_y = (src_h - 1) as f32;

    for y in y0..y1 {
        for x in x0..x1 {
            let dx = x as f32 + 0.5;
            let dy = y as f32 + 0.5;
            let (sx, sy) = match project(&viewport_to_bitmap, dx, dy) {
                Some(p) => p,
                None => continue,
            };
            if sx < 0.0 || sy < 0.0 || sx > src_max_x || sy > src_max_y {
                continue;
            }
            let x0_i = sx.floor() as u32;
            let y0_i = sy.floor() as u32;
            let x1_i = (x0_i + 1).min(src_w - 1);
            let y1_i = (y0_i + 1).min(src_h - 1);
            let fx = sx - x0_i as f32;
            let fy = sy - y0_i as f32;
            let i_tl = ((y0_i * src_w + x0_i) * 4) as usize;
            let i_tr = ((y0_i * src_w + x1_i) * 4) as usize;
            let i_bl = ((y1_i * src_w + x0_i) * 4) as usize;
            let i_br = ((y1_i * src_w + x1_i) * 4) as usize;
            let a = bilinear_u8(
                src[i_tl + 3],
                src[i_tr + 3],
                src[i_bl + 3],
                src[i_br + 3],
                fx,
                fy,
            );
            if a == 0 {
                continue;
            }
            let r = bilinear_u8(src[i_tl], src[i_tr], src[i_bl], src[i_br], fx, fy);
            let g = bilinear_u8(
                src[i_tl + 1],
                src[i_tr + 1],
                src[i_bl + 1],
                src[i_br + 1],
                fx,
                fy,
            );
            let b = bilinear_u8(
                src[i_tl + 2],
                src[i_tr + 2],
                src[i_bl + 2],
                src[i_br + 2],
                fx,
                fy,
            );
            let dst_idx = ((y * dst_w + x) * 4) as usize;
            blend_source_over(&mut dst[dst_idx..dst_idx + 4], [r, g, b, a]);
        }
    }
}

#[inline]
fn bilinear_u8(tl: u8, tr: u8, bl: u8, br: u8, fx: f32, fy: f32) -> u8 {
    let t = (tl as f32) * (1.0 - fx) + (tr as f32) * fx;
    let b = (bl as f32) * (1.0 - fx) + (br as f32) * fx;
    let v = t * (1.0 - fy) + b * fy;
    v.round().clamp(0.0, 255.0) as u8
}

/// Source-over alpha composite: `dst = src + dst * (1 - src_a)`.
/// Output alpha is forced to 255 because the destination buffer is a
/// solid camera frame; no further compositing happens above us.
#[inline]
fn blend_source_over(dst: &mut [u8], src: [u8; 4]) {
    let sa = src[3] as u32;
    if sa == 0 {
        return;
    }
    if sa == 255 {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
        return;
    }
    let inv = 255 - sa;
    dst[0] = ((src[0] as u32 * sa + dst[0] as u32 * inv) / 255) as u8;
    dst[1] = ((src[1] as u32 * sa + dst[1] as u32 * inv) / 255) as u8;
    dst[2] = ((src[2] as u32 * sa + dst[2] as u32 * inv) / 255) as u8;
    dst[3] = 255;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[r, g, b, 255]);
        }
        v
    }

    #[test]
    fn rotation_zero_copies_camera_through() {
        let cam = solid_rgba(2, 3, 10, 20, 30);
        let mut dst = vec![0u8; 24];
        composite_frame_into(&mut dst, 2, 3, &cam, 2, 3, 0, &IDENTITY_H, &[]).unwrap();
        assert_eq!(dst, cam);
    }

    #[test]
    fn rotation_90_rotates_pixel_layout() {
        // 2x1 source: (red, green) → 1x2 destination after 90° CW: (red at bottom, green at top).
        let mut cam = Vec::new();
        cam.extend_from_slice(&[255, 0, 0, 255]); // (0,0)
        cam.extend_from_slice(&[0, 255, 0, 255]); // (1,0)
        let mut dst = vec![0u8; 8];
        composite_frame_into(&mut dst, 1, 2, &cam, 2, 1, 90, &IDENTITY_H, &[]).unwrap();
        // dst[0][0] = src[1 - 0 - 0][0] = src[0,0] is wrong — with our mapping
        // dst[y][x] = src[src_h - 1 - x][y]. For dst[0][0]: src[0][0] = red.
        // For dst[1][0]: src[0][1] = src[0,1] = green.
        // Wait: src_h = 1 so src_h - 1 - x = 0 - x = -x, always 0 for x=0.
        // dst[0][0] = src[src_h-1-0][0] = src[0,0] = (255,0,0,255). ✓
        // dst[1][0] = src[src_h-1-0][1] = src[0,1] = (0,255,0,255). ✓
        assert_eq!(&dst[0..4], &[255, 0, 0, 255]);
        assert_eq!(&dst[4..8], &[0, 255, 0, 255]);
    }

    #[test]
    fn overlay_with_identity_h_lands_at_surface_origin() {
        // 6x6 camera. Opaque uniform-red overlay 4x4 at surface
        // origin (1, 1). With identity H, the overlay should occupy
        // viewport pixels [1..5) × [1..5). Pixels outside should keep
        // the camera grey.
        let cam = solid_rgba(6, 6, 50, 50, 50);
        let overlay = solid_rgba(4, 4, 200, 0, 0);
        let mut dst = vec![0u8; 6 * 6 * 4];
        let item = OverlayItem {
            bitmap_rgba: &overlay,
            bitmap_width: 4,
            bitmap_height: 4,
            bitmap_origin_surface_x: 1.0,
            bitmap_origin_surface_y: 1.0,
        };
        composite_frame_into(
            &mut dst,
            6,
            6,
            &cam,
            6,
            6,
            0,
            &IDENTITY_H,
            std::slice::from_ref(&item),
        )
        .unwrap();
        // Centre of overlay region: pixel (2, 2). Uniform-red overlay
        // → bilinear sampling still gives pure red.
        let inside = ((2 * 6 + 2) * 4) as usize;
        assert_eq!(&dst[inside..inside + 3], &[200, 0, 0]);
        // Camera-only corner: pixel (0, 0).
        assert_eq!(&dst[0..3], &[50, 50, 50]);
        // Camera-only outside the overlay region: pixel (5, 5).
        let outside = ((5 * 6 + 5) * 4) as usize;
        assert_eq!(&dst[outside..outside + 3], &[50, 50, 50]);
    }

    const IDENTITY_H: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
}

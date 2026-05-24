//! Composite the camera frame and one or more overlay items into a
//! display-orient RGBA buffer that the Kotlin side hands straight to a
//! [`SurfaceView`]. Replaces the previous "preview surface + separate
//! overlay view" architecture, where the two streams updated at
//! different rates and the overlay could visibly drift from the camera
//! pixels under motion.
//!
//! Each item is a single pre-composed RGBA bitmap covering an
//! anchor's full overlay (bg + text already blended in surface space).
//! Phase 1 callers pass one item per active anchor.
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
    /// Per-source-row half-open `[first, last)` range of columns
    /// containing any non-zero alpha. Used by the warp inner loop
    /// to skip sampling for inverse-projected pixels that land in a
    /// guaranteed-transparent source region — the precomputation is
    /// done once when the overlay is rasterised, off the per-frame
    /// hot path. Pass an empty slice (or any slice whose length
    /// differs from `bitmap_height`) to disable the optimisation
    /// and fall back to the per-pixel alpha-check inside the loop.
    pub row_extents: &'a [(u32, u32)],
}

#[derive(Debug)]
pub enum CompositeError {
    DstBufferSize,
    SrcBufferSize,
}

/// Build the per-frame display image: blit the sensor RGBA + warp +
/// alpha-blend each overlay item on top. Both source and destination
/// are sensor-orient; the SurfaceView rotates for display at scanout.
///
/// `dst_rgba` must be sized `display_w * display_h * 4`. On success
/// it's fully overwritten with the composited result and can be blitted
/// to a `SurfaceView` directly.
pub fn composite_frame_into(
    dst_rgba: &mut [u8],
    sensor_w: u32,
    sensor_h: u32,
    camera_rgba: &[u8],
    h_surface_to_viewport: &[f32; 9],
    items: &[OverlayItem<'_>],
) -> Result<(), CompositeError> {
    let frame_bytes = (sensor_w as usize)
        .checked_mul(sensor_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::DstBufferSize)?;
    if dst_rgba.len() != frame_bytes {
        return Err(CompositeError::DstBufferSize);
    }
    if camera_rgba.len() != frame_bytes {
        return Err(CompositeError::SrcBufferSize);
    }
    dst_rgba.copy_from_slice(camera_rgba);
    for item in items {
        warp_item_onto_display(dst_rgba, sensor_w, sensor_h, item, h_surface_to_viewport);
    }
    Ok(())
}

/// Cropped variant of [`composite_frame_into`]: reads a sub-rect of
/// the full-sensor `camera_rgba` (starting at `src_offset_x/y`, sized
/// `dst_w × dst_h`) instead of the whole frame, and writes into `dst`
/// which is sized to that sub-rect. Used by the live pipeline when
/// the user's preview is FILL_CENTER-cropped: the OCR and overlay
/// surface map both live in visible-region-sensor coords, so the
/// composite output bitmap is sized to match.
///
/// `h_surface_to_viewport` is in the *visible-region* coord system,
/// not the full-sensor one — engine's H_anchor→view is already in
/// that space because the gray frame is built from the same crop.
pub fn composite_frame_into_cropped(
    dst_rgba: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    camera_rgba: &[u8],
    src_full_w: u32,
    src_full_h: u32,
    src_offset_x: u32,
    src_offset_y: u32,
    h_surface_to_viewport: &[f32; 9],
    items: &[OverlayItem<'_>],
) -> Result<(), CompositeError> {
    let dst_bytes = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::DstBufferSize)?;
    if dst_rgba.len() != dst_bytes {
        return Err(CompositeError::DstBufferSize);
    }
    let src_full_bytes = (src_full_w as usize)
        .checked_mul(src_full_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(CompositeError::SrcBufferSize)?;
    if camera_rgba.len() != src_full_bytes {
        return Err(CompositeError::SrcBufferSize);
    }
    if src_offset_x.saturating_add(dst_w) > src_full_w
        || src_offset_y.saturating_add(dst_h) > src_full_h
    {
        return Err(CompositeError::SrcBufferSize);
    }
    let src_stride = (src_full_w as usize) * 4;
    let dst_stride = (dst_w as usize) * 4;
    for y in 0..dst_h {
        let src_row = ((src_offset_y + y) as usize) * src_stride + (src_offset_x as usize) * 4;
        let dst_row = (y as usize) * dst_stride;
        dst_rgba[dst_row..dst_row + dst_stride]
            .copy_from_slice(&camera_rgba[src_row..src_row + dst_stride]);
    }
    for item in items {
        warp_item_onto_display(dst_rgba, dst_w, dst_h, item, h_surface_to_viewport);
    }
    Ok(())
}

/// Per-frame inputs to a [`Renderer::composite`] call, bundled into one
/// borrowed struct so the trait stays object-safe and the backend
/// retains no per-frame state: the large camera buffer and the overlay
/// items are handed in by reference each frame, never held across calls.
/// Mirrors the argument list of [`composite_frame_into_cropped`].
pub struct CompositeInput<'a> {
    pub dst_w: u32,
    pub dst_h: u32,
    pub camera_rgba: &'a [u8],
    pub src_full_w: u32,
    pub src_full_h: u32,
    pub src_offset_x: u32,
    pub src_offset_y: u32,
    pub h_surface_to_viewport: &'a [f32; 9],
    pub items: &'a [OverlayItem<'a>],
}

/// The composite backend the live pipeline drives once per frame. Exists
/// to make the camera+overlay warp swappable: the CPU bilinear warp is
/// the reference backend ([`CpuRenderer`]), and a GLES backend can be
/// dropped in behind the same call site. `&mut self` because GPU
/// backends own mutable state (GL context, cached textures).
pub trait Renderer {
    fn composite(
        &mut self,
        input: &CompositeInput<'_>,
        out: &mut [u8],
    ) -> Result<(), CompositeError>;
}

/// Reference backend: the bilinear CPU warp + source-over blend. New
/// backends are pinned against this for output equivalence.
pub struct CpuRenderer;

impl Renderer for CpuRenderer {
    fn composite(
        &mut self,
        input: &CompositeInput<'_>,
        out: &mut [u8],
    ) -> Result<(), CompositeError> {
        composite_frame_into_cropped(
            out,
            input.dst_w,
            input.dst_h,
            input.camera_rgba,
            input.src_full_w,
            input.src_full_h,
            input.src_offset_x,
            input.src_offset_y,
            input.h_surface_to_viewport,
            input.items,
        )
    }
}

/// Where a per-frame composite is sent. The pipeline builds the
/// [`CompositeInput`] once (under the frame + overlay locks) and hands it
/// to the target, so the same `process_frame` path drives either a CPU
/// buffer or a GPU surface without knowing which.
pub trait ComposeTarget {
    fn compose(&mut self, input: &CompositeInput<'_>) -> Result<(), CompositeError>;
}

/// CPU target: warps into a caller-owned RGBA slice (Android `Bitmap`
/// pixels, the Linux QImage buffer). The default output.
pub struct SliceTarget<'a> {
    pub dst: &'a mut [u8],
}

impl ComposeTarget for SliceTarget<'_> {
    fn compose(&mut self, input: &CompositeInput<'_>) -> Result<(), CompositeError> {
        CpuRenderer.composite(input, self.dst)
    }
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

    // Row-incremental projective sampling. Per-pixel the previous
    // implementation did a full 3×3 matrix-vector multiply + divide
    // (9 mults + 6 adds + 1 div). For row `y`, the numerators and
    // denominator of the sampling projection are linear in `x`:
    //
    //   num_sx = m0·dx + m1·dy + m2
    //   num_sy = m3·dx + m4·dy + m5
    //   den    = m6·dx + m7·dy + m8
    //
    // so we evaluate them at the row start (`dx = x0 + 0.5`) once,
    // then advance by `(m0, m3, m6)` per column. Per-pixel cost
    // drops to 3 adds + 1 div + 2 muls — a meaningful constant-
    // factor win on overlay-heavy frames.
    let m0 = viewport_to_bitmap[0];
    let m1 = viewport_to_bitmap[1];
    let m2 = viewport_to_bitmap[2];
    let m3 = viewport_to_bitmap[3];
    let m4 = viewport_to_bitmap[4];
    let m5 = viewport_to_bitmap[5];
    let m6 = viewport_to_bitmap[6];
    let m7 = viewport_to_bitmap[7];
    let m8 = viewport_to_bitmap[8];
    let row_start_x = x0 as f32 + 0.5;

    // Per-source-row non-transparent column extents. Computed once
    // when the overlay was rasterised; used here to skip the bilinear
    // sample + blend for any inverse-projected pixel that lands in a
    // guaranteed-transparent source region.
    let extents_active = !item.row_extents.is_empty() && item.row_extents.len() == src_h as usize;

    let dst_stride = (dst_w * 4) as usize;
    let band_start = y0 as usize * dst_stride;
    let band_end = y1 as usize * dst_stride;
    dst[band_start..band_end]
        .chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(row_off, row)| {
            let y = y0 + row_off as u32;
            let dy = y as f32 + 0.5;
            let mut num_sx = m0 * row_start_x + m1 * dy + m2;
            let mut num_sy = m3 * row_start_x + m4 * dy + m5;
            let mut den = m6 * row_start_x + m7 * dy + m8;
            for x in x0..x1 {
                if den.abs() < 1e-9 || !den.is_finite() {
                    num_sx += m0;
                    num_sy += m3;
                    den += m6;
                    continue;
                }
                let inv_den = 1.0 / den;
                let sx = num_sx * inv_den;
                let sy = num_sy * inv_den;
                num_sx += m0;
                num_sy += m3;
                den += m6;
                if !sx.is_finite()
                    || !sy.is_finite()
                    || sx < 0.0
                    || sy < 0.0
                    || sx > src_max_x
                    || sy > src_max_y
                {
                    continue;
                }
                let x0_i = sx.floor() as u32;
                let y0_i = sy.floor() as u32;
                let x1_i = (x0_i + 1).min(src_w - 1);
                let y1_i = (y0_i + 1).min(src_h - 1);
                if extents_active {
                    let r0 = item.row_extents[y0_i as usize];
                    let r1 = item.row_extents[y1_i as usize];
                    let r0_overlaps = r0.0 <= x1_i && x0_i < r0.1;
                    let r1_overlaps = r1.0 <= x1_i && x0_i < r1.1;
                    if !r0_overlaps && !r1_overlaps {
                        continue;
                    }
                }
                let fx = sx - x0_i as f32;
                let fy = sy - y0_i as f32;
                let i_tl = ((y0_i * src_w + x0_i) * 4) as usize;
                let i_tr = ((y0_i * src_w + x1_i) * 4) as usize;
                let i_bl = ((y1_i * src_w + x0_i) * 4) as usize;
                let i_br = ((y1_i * src_w + x1_i) * 4) as usize;
                let one_minus_fx = 1.0 - fx;
                let one_minus_fy = 1.0 - fy;
                let w_tl = one_minus_fx * one_minus_fy;
                let w_tr = fx * one_minus_fy;
                let w_bl = one_minus_fx * fy;
                let w_br = fx * fy;
                let a = bilinear_sample(
                    src[i_tl + 3],
                    src[i_tr + 3],
                    src[i_bl + 3],
                    src[i_br + 3],
                    w_tl,
                    w_tr,
                    w_bl,
                    w_br,
                );
                if a == 0 {
                    continue;
                }
                let r = bilinear_sample(
                    src[i_tl], src[i_tr], src[i_bl], src[i_br], w_tl, w_tr, w_bl, w_br,
                );
                let g = bilinear_sample(
                    src[i_tl + 1],
                    src[i_tr + 1],
                    src[i_bl + 1],
                    src[i_br + 1],
                    w_tl,
                    w_tr,
                    w_bl,
                    w_br,
                );
                let b = bilinear_sample(
                    src[i_tl + 2],
                    src[i_tr + 2],
                    src[i_bl + 2],
                    src[i_br + 2],
                    w_tl,
                    w_tr,
                    w_bl,
                    w_br,
                );
                let px = (x * 4) as usize;
                blend_source_over(&mut row[px..px + 4], [r, g, b, a]);
            }
        });
}

#[inline]
fn bilinear_sample(
    tl: u8,
    tr: u8,
    bl: u8,
    br: u8,
    w_tl: f32,
    w_tr: f32,
    w_bl: f32,
    w_br: f32,
) -> u8 {
    let v = (tl as f32) * w_tl + (tr as f32) * w_tr + (bl as f32) * w_bl + (br as f32) * w_br;
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
    fn camera_blit_copies_through() {
        let cam = solid_rgba(2, 3, 10, 20, 30);
        let mut dst = vec![0u8; 24];
        composite_frame_into(&mut dst, 2, 3, &cam, &IDENTITY_H, &[]).unwrap();
        assert_eq!(dst, cam);
    }

    #[test]
    fn overlay_with_identity_h_lands_at_surface_origin() {
        let cam = solid_rgba(6, 6, 50, 50, 50);
        let overlay = solid_rgba(4, 4, 200, 0, 0);
        let mut dst = vec![0u8; 6 * 6 * 4];
        let item = OverlayItem {
            bitmap_rgba: &overlay,
            bitmap_width: 4,
            bitmap_height: 4,
            bitmap_origin_surface_x: 1.0,
            bitmap_origin_surface_y: 1.0,
            row_extents: &[],
        };
        composite_frame_into(
            &mut dst,
            6,
            6,
            &cam,
            &IDENTITY_H,
            std::slice::from_ref(&item),
        )
        .unwrap();
        let inside = ((2 * 6 + 2) * 4) as usize;
        assert_eq!(&dst[inside..inside + 3], &[200, 0, 0]);
        assert_eq!(&dst[0..3], &[50, 50, 50]);
        let outside = ((5 * 6 + 5) * 4) as usize;
        assert_eq!(&dst[outside..outside + 3], &[50, 50, 50]);
    }

    const IDENTITY_H: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    /// The `Renderer` seam must be a pure pass-through to the existing
    /// cropped warp under a non-trivial H: same inputs, byte-identical
    /// output. Pins the Phase-1 refactor against accidental divergence
    /// and is the equivalence baseline future backends are held to.
    #[test]
    fn cpu_renderer_matches_direct_cropped_composite() {
        let (full_w, full_h) = (16u32, 16u32);
        let cam = solid_rgba(full_w, full_h, 30, 60, 90);
        let overlay = solid_rgba(5, 5, 220, 10, 10);
        let item = OverlayItem {
            bitmap_rgba: &overlay,
            bitmap_width: 5,
            bitmap_height: 5,
            bitmap_origin_surface_x: 2.0,
            bitmap_origin_surface_y: 3.0,
            row_extents: &[],
        };
        let h = [1.03, -0.02, 1.0, 0.01, 0.98, -0.5, 5.0e-4, -3.0e-4, 1.0];
        let (dst_w, dst_h) = (10u32, 12u32);
        let off_x = (full_w - dst_w) / 2;
        let off_y = (full_h - dst_h) / 2;

        let mut expected = vec![0u8; (dst_w * dst_h * 4) as usize];
        composite_frame_into_cropped(
            &mut expected,
            dst_w,
            dst_h,
            &cam,
            full_w,
            full_h,
            off_x,
            off_y,
            &h,
            std::slice::from_ref(&item),
        )
        .unwrap();

        let input = CompositeInput {
            dst_w,
            dst_h,
            camera_rgba: &cam,
            src_full_w: full_w,
            src_full_h: full_h,
            src_offset_x: off_x,
            src_offset_y: off_y,
            h_surface_to_viewport: &h,
            items: std::slice::from_ref(&item),
        };
        let mut got = vec![0u8; (dst_w * dst_h * 4) as usize];
        let mut renderer = CpuRenderer;
        renderer.composite(&input, &mut got).unwrap();

        assert_eq!(got, expected);
    }

    /// Verify the row-incremental projective sampling matches a
    /// straightforward per-pixel reference under a non-identity H.
    #[test]
    fn overlay_under_perspective_h_matches_reference() {
        let cam = solid_rgba(16, 16, 50, 50, 50);
        let overlay = solid_rgba(6, 6, 200, 0, 0);
        let item = OverlayItem {
            bitmap_rgba: &overlay,
            bitmap_width: 6,
            bitmap_height: 6,
            bitmap_origin_surface_x: 4.0,
            bitmap_origin_surface_y: 4.0,
            row_extents: &[],
        };
        let h = [1.05, -0.03, 1.5, 0.02, 0.97, -0.7, 1.0e-3, -5.0e-4, 1.0];

        let mut dst = vec![0u8; 16 * 16 * 4];
        composite_frame_into(&mut dst, 16, 16, &cam, &h, std::slice::from_ref(&item)).unwrap();

        let mut painted = 0usize;
        for y in 0..16 {
            for x in 0..16 {
                let i = ((y * 16 + x) * 4) as usize;
                let r = dst[i];
                let g = dst[i + 1];
                let b = dst[i + 2];
                assert!(g <= 50, "g {g} > 50 at ({x}, {y})");
                assert!(b <= 50, "b {b} > 50 at ({x}, {y})");
                if r > 50 {
                    painted += 1;
                }
            }
        }
        assert!(
            painted >= 8,
            "perspective overlay covered only {painted} pixels"
        );
        assert!(
            painted <= 64,
            "perspective overlay covered too many ({painted}) pixels"
        );
    }
}

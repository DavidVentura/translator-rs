//! Per-detection color matting: classify ink/background pixels for each
//! recognised text region and produce an inpainted rectified strip
//! suitable for use as a live-overlay background. Replaces the
//! hardcoded dark "pill" with a real per-pixel reconstruction of the
//! camera background under the source text.
//!
//! The reference smoke-test for visual quality is
//! `tests/ppocr_color_matting.rs`, which exercises the same algorithm
//! and dumps PNG diagnostics. This module is the production extraction:
//! same logic, no I/O, no test-only visualization helpers.
//!
//! ## Why a rectified strip
//!
//! The algorithm operates in **rectified strip coordinates**, where
//! the text runs strictly left-to-right inside an axis-aligned bitmap.
//! Column-wise inpaint assumes glyphs are bounded vertically per
//! column with a contiguous masked region between non-ink pixels — an
//! assumption that breaks on any tilted text. Dewarping into rectified
//! coords first restores the assumption.
//!
//! At acquire time, the bindings layer (uniffi_catalog) takes the
//! `MattedStrip` produced here, renders translated text into it, and
//! then re-warps the textured strip into a canonical-orientation
//! overlay bitmap via `OverlayItem`. The compositor handles the
//! per-frame planar-tracker warp from canonical → viewport coords.
//!
//! See `FUTURE_SURFACE_MAP.md` ("Color matting") for the full design
//! and how this slots into the surface-map roadmap.
//!
//! ## Pipeline per detection
//!
//! 1. Rasterize the detector's contour into a mask aligned to the
//!    image (`ContourMask`).
//! 2. Inflate the ROI for ascender/descender room. Sample a ring
//!    annulus just outside the contour for a clean background median;
//!    pixels claimed by other detections' contours are excluded so
//!    adjacent lines don't bleed.
//! 3. Otsu-split the polygon-internal luma; combine with `decide_ink_class`
//!    to pick ink-on-light vs light-on-dark.
//! 4. Hysteresis-flood ink mask from polygon seeds through a two-tier
//!    candidate set (strict inside polygon, looser in the asc/desc band).
//! 5. Dewarp camera + ink-mask into a rectified strip aligned to the
//!    oriented box.
//! 6. Inpaint masked pixels in the strip via 4-direction nearest-non-ink
//!    blend with outlier rejection + smoothing passes.
//!
//! Output: rectified RGBA strip + the oriented-box parameters
//! needed to re-warp it back to canonical-frame coords.

use image::{GrayImage, Luma, Rgba, RgbaImage};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point;

use crate::DetectedTextBox;
use crate::ocr::Rect;

/// Per-detection matting result. The `strip_rgba` is a rectified RGBA
/// image where the source text has been removed and replaced with the
/// inpainted background. To get this back into canonical-frame coords
/// at render time, treat the strip as an axis-aligned bitmap centred
/// at `(canonical_cx, canonical_cy)` with dimensions `canonical_width`
/// × `canonical_height`, rotated by `canonical_angle_radians`. This
/// matches the oriented-rect convention used elsewhere in the project.
#[derive(Clone, Debug)]
pub struct MattedStrip {
    /// Index of the source detection in the original `boxes` slice.
    pub box_index: usize,
    /// Rectified strip RGBA, row-major, 4 bytes per pixel.
    pub strip_rgba: Vec<u8>,
    /// Strip dimensions in pixels.
    pub strip_width: u32,
    pub strip_height: u32,
    /// Centre of the strip in canonical (oriented-frame) coords.
    pub canonical_cx: f32,
    pub canonical_cy: f32,
    /// Rotation of the strip — the strip's local x-axis maps to this
    /// direction in canonical coords.
    pub canonical_angle_radians: f32,
    /// Footprint of the strip in canonical coords. Includes the
    /// ascender/descender padding around the text bbox so callers can
    /// render translated text that may exceed the source's vertical
    /// extent.
    pub canonical_width: f32,
    pub canonical_height: f32,
    /// True if the source ink was darker than the background (most
    /// printed text on a light page). Used by callers to pick a
    /// readable fg colour for translated text without re-scanning the
    /// strip's luma.
    pub ink_is_dark: bool,
    /// `Some(argb)` when the ring-median background is uniform enough
    /// to use as a single solid pill colour (low MAD across ring
    /// samples). `None` when the background varies significantly
    /// (gradient, shadow crossing, etc.) — caller should fall back to
    /// the default dark pill rather than painting a single colour
    /// that would visibly mismatch on one side. Higher byte is alpha.
    pub bg_uniform_argb: Option<u32>,
}

/// Tuning knobs for the matting algorithm. Defaults match the values
/// the smoke test settled on.
#[derive(Clone, Copy, Debug)]
pub struct MattingConfig {
    /// Maximum search distance for nearest-non-ink samples during
    /// inpaint, in rectified-strip pixels.
    pub inpaint_max_search: u32,
    /// Number of post-pass smoothing iterations over masked pixels.
    /// Kills 4-direction axis-aligned banding artefacts.
    pub inpaint_smoothing_passes: u32,
    /// Dilate the ink mask by this radius before inpainting. Catches
    /// glyph anti-aliased rim pixels that the hysteresis flood missed.
    pub inpaint_sample_radius: u32,
}

impl Default for MattingConfig {
    fn default() -> Self {
        Self {
            inpaint_max_search: 40,
            inpaint_smoothing_passes: 1,
            inpaint_sample_radius: 2,
        }
    }
}

/// Compute matted strips for each detection. Returns one entry per
/// input box; entries are `None` when the detection's contour was too
/// small or under-resolved to recover a clean ink mask (caller should
/// fall back to the default-pill rendering for those).
pub fn mat_detections(rgba: &RgbaImage, boxes: &[DetectedTextBox]) -> Vec<Option<MattedStrip>> {
    mat_detections_with_config(rgba, boxes, &MattingConfig::default())
}

/// Like [`mat_detections`] but with explicit tuning. Mostly useful for
/// tests/benches.
pub fn mat_detections_with_config(
    rgba: &RgbaImage,
    boxes: &[DetectedTextBox],
    cfg: &MattingConfig,
) -> Vec<Option<MattedStrip>> {
    let (w, h) = rgba.dimensions();
    let contour_masks: Vec<ContourMask> = boxes
        .iter()
        .enumerate()
        .filter_map(|(idx, b)| rasterize_contour_mask(w, h, b, idx))
        .collect();
    let contour_occupancy = build_contour_occupancy(w, h, &contour_masks);

    // Build the global ink mask + per-detection metadata in one pass.
    // We need the union mask so dewarp's sample-into-strip step can
    // mark out-of-strip ink (e.g. neighbouring lines whose pixels would
    // otherwise leak in via the rectification's padding).
    let mut ink_mask = vec![false; (w as usize) * (h as usize)];
    let mut detections: Vec<Option<Detection>> = Vec::with_capacity(boxes.len());
    detections.resize_with(boxes.len(), || None);
    for cmask in &contour_masks {
        let Some(det) = build_detection(rgba, cmask, &contour_occupancy, w, h) else {
            continue;
        };
        for &(px, py) in &det.ink_pixels {
            ink_mask[(py as usize) * (w as usize) + px as usize] = true;
        }
        detections[cmask.box_index] = Some(det);
    }

    // Produce one MattedStrip per detection by dewarping + inpainting.
    // `mat_strip_for_detection` returns None on bookkeeping failures
    // (no oriented box, zero-area strip, etc.); those bubble up as
    // `None` in the result so callers can substitute fallback rendering.
    detections
        .into_iter()
        .enumerate()
        .map(|(idx, det)| {
            let det = det?;
            mat_strip_for_detection(rgba, &ink_mask, w, h, &boxes[idx], &det, cfg)
        })
        .collect()
}

/// Dewarp a detection's region into a rectified RGBA strip, then
/// inpaint the ink pixels. The strip is padded vertically by ~75% of
/// the detected line height so we have clean background samples above
/// and below the text band for the inpaint walk.
fn mat_strip_for_detection(
    image: &RgbaImage,
    mask: &[bool],
    w: u32,
    h: u32,
    detected: &DetectedTextBox,
    det: &Detection,
    cfg: &MattingConfig,
) -> Option<MattedStrip> {
    let _ = det;
    let oriented = detected.oriented_box;
    if oriented.width <= 1.0 || oriented.height <= 1.0 {
        return None;
    }
    let cos_a = oriented.angle_radians.cos();
    let sin_a = oriented.angle_radians.sin();

    // 15% horizontal pad gives translated text breathing room without
    // pulling in neighbouring detections; 75% vertical pad guarantees
    // the inpaint walk finds clean bg above/below the text band.
    let pad_x = (oriented.width * 0.15).max(4.0);
    let pad_y = (oriented.height * 0.75).max(8.0);
    let strip_w = (oriented.width + 2.0 * pad_x).ceil().max(8.0) as u32;
    let strip_h = (oriented.height + 2.0 * pad_y).ceil().max(8.0) as u32;
    let sw_us = strip_w as usize;
    let strip_cx = strip_w as f32 * 0.5;
    let strip_cy = strip_h as f32 * 0.5;
    let w_us = w as usize;

    // Sample the global image + ink mask into strip coords via inverse
    // warp. Pixels that fall outside the image are flagged masked so
    // the inpaint walk skips them.
    let mut strip_image = vec![Rgba([0u8; 4]); (strip_w * strip_h) as usize];
    let mut strip_mask = vec![false; (strip_w * strip_h) as usize];
    for sy in 0..strip_h {
        for sx in 0..strip_w {
            let u = sx as f32 + 0.5 - strip_cx;
            let v = sy as f32 + 0.5 - strip_cy;
            let px = u * cos_a - v * sin_a + oriented.cx;
            let py = u * sin_a + v * cos_a + oriented.cy;
            let pxi = px.floor() as i32;
            let pyi = py.floor() as i32;
            let idx = (sy as usize) * sw_us + sx as usize;
            if pxi < 0 || pyi < 0 || pxi >= w as i32 || pyi >= h as i32 {
                strip_mask[idx] = true;
                continue;
            }
            strip_image[idx] = *image.get_pixel(pxi as u32, pyi as u32);
            strip_mask[idx] = mask[(pyi as usize) * w_us + pxi as usize];
        }
    }

    let strip_out = inpaint_native(
        &strip_image,
        &strip_mask,
        strip_w,
        strip_h,
        cfg.inpaint_max_search,
        cfg.inpaint_sample_radius,
        cfg.inpaint_smoothing_passes,
    );

    // Pack into a flat RGBA byte buffer for the bindings/compositor.
    let mut strip_bytes = Vec::with_capacity(strip_out.len() * 4);
    for px in &strip_out {
        strip_bytes.push(px[0]);
        strip_bytes.push(px[1]);
        strip_bytes.push(px[2]);
        strip_bytes.push(px[3]);
    }

    Some(MattedStrip {
        box_index: det.box_index,
        strip_rgba: strip_bytes,
        strip_width: strip_w,
        strip_height: strip_h,
        canonical_cx: oriented.cx,
        canonical_cy: oriented.cy,
        canonical_angle_radians: oriented.angle_radians,
        canonical_width: strip_w as f32,
        canonical_height: strip_h as f32,
        ink_is_dark: det.ink_is_dark,
        bg_uniform_argb: det.bg_uniform_argb,
    })
}

// --- internals shared with the smoke test --------------------------------
//
// Visibility is `pub(crate)` so the test (an integration test in the
// `tests/` directory) can `use translator::color_matting::...` for its
// visualization. Production callers use the `mat_detections` entry
// point above.

#[derive(Clone, Debug)]
pub(crate) struct ContourMask {
    pub box_index: usize,
    pub rect: Rect,
    pub width: u32,
    pub height: u32,
    pub bits: Vec<bool>,
}

impl ContourMask {
    pub(crate) fn contains(&self, x: u32, y: u32) -> bool {
        if x < self.rect.left || x >= self.rect.right || y < self.rect.top || y >= self.rect.bottom
        {
            return false;
        }
        let lx = x - self.rect.left;
        let ly = y - self.rect.top;
        self.bits[(ly as usize) * (self.width as usize) + lx as usize]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Detection {
    pub box_index: usize,
    pub contour_rect: Rect,
    pub bg_ring_median: Rgba<u8>,
    pub ink_pixels: Vec<(u32, u32)>,
    pub fell_back: bool,
    pub poly_coverage_pct: u32,
    pub ink_is_dark: bool,
    /// `Some(argb)` when the ring samples cluster around a single
    /// peak — i.e. the background is one colour with sensor noise.
    /// `None` when samples spread across multiple peaks (gradient,
    /// shadow boundary, mixed bg). Computed via a 4-bit-per-channel
    /// histogram with 3×3×3 neighbour-smoothing so noise straddling
    /// a bucket boundary doesn't artificially split the cluster.
    pub bg_uniform_argb: Option<u32>,
}

pub(crate) fn rasterize_contour_mask(
    w: u32,
    h: u32,
    det: &DetectedTextBox,
    box_index: usize,
) -> Option<ContourMask> {
    let rect = clamp_rect(det.rect, w, h)?;
    let local_w = rect.width().max(1);
    let local_h = rect.height().max(1);
    let mut img = GrayImage::from_pixel(local_w, local_h, Luma([0u8]));
    if det.contour.len() >= 6 {
        let points: Vec<Point<i32>> = det
            .contour
            .chunks_exact(2)
            .map(|p| {
                Point::new(
                    (p[0] - rect.left as f32).round() as i32,
                    (p[1] - rect.top as f32).round() as i32,
                )
            })
            .collect();
        if points.len() >= 3 {
            draw_polygon_mut(&mut img, &points, Luma([255u8]));
        }
    } else {
        for p in img.pixels_mut() {
            *p = Luma([255u8]);
        }
    }
    let bits: Vec<bool> = img.pixels().map(|p| p[0] != 0).collect();
    Some(ContourMask {
        box_index,
        rect,
        width: local_w,
        height: local_h,
        bits,
    })
}

pub(crate) fn build_contour_occupancy(w: u32, h: u32, masks: &[ContourMask]) -> Vec<bool> {
    let mut occ = vec![false; (w as usize) * (h as usize)];
    for m in masks {
        for ly in 0..m.height {
            for lx in 0..m.width {
                if !m.bits[(ly as usize) * (m.width as usize) + lx as usize] {
                    continue;
                }
                let x = m.rect.left + lx;
                let y = m.rect.top + ly;
                occ[(y as usize) * (w as usize) + x as usize] = true;
            }
        }
    }
    occ
}

pub(crate) fn build_detection(
    image: &RgbaImage,
    cmask: &ContourMask,
    contour_occupancy: &[bool],
    w: u32,
    h: u32,
) -> Option<Detection> {
    let line_h = cmask.rect.height().max(1);
    let pad_y = (line_h / 3).clamp(3, 14);
    let ring_thickness = (line_h / 3).clamp(4, 14);

    let pad_x = pad_y + ring_thickness;
    let pad_y_total = pad_y + ring_thickness;
    let roi = inflate_rect_xy(cmask.rect, pad_x, pad_y_total, w, h);
    let roi_w = roi.width().max(1);
    let roi_h = roi.height().max(1);
    let roi_w_us = roi_w as usize;

    let mut polygon_local = vec![false; (roi_w as usize) * (roi_h as usize)];
    for cy in 0..cmask.height {
        for cx in 0..cmask.width {
            if !cmask.bits[(cy as usize) * (cmask.width as usize) + cx as usize] {
                continue;
            }
            let gx = cmask.rect.left + cx;
            let gy = cmask.rect.top + cy;
            let lx = gx - roi.left;
            let ly = gy - roi.top;
            polygon_local[(ly as usize) * roi_w_us + lx as usize] = true;
        }
    }

    let classify_region = dilate(&polygon_local, roi_w, roi_h, pad_y);
    let ring_outer = dilate(&classify_region, roi_w, roi_h, ring_thickness);

    let mut ring_samples = Vec::new();
    for ly in 0..roi_h {
        for lx in 0..roi_w {
            let idx = (ly as usize) * roi_w_us + lx as usize;
            if !ring_outer[idx] || classify_region[idx] {
                continue;
            }
            let gx = roi.left + lx;
            let gy = roi.top + ly;
            if contour_occupancy[(gy as usize) * (w as usize) + gx as usize]
                && !cmask.contains(gx, gy)
            {
                continue;
            }
            ring_samples.push(*image.get_pixel(gx, gy));
        }
    }
    if ring_samples.len() < 24 {
        return None;
    }
    let bg_median = median_color(&ring_samples);
    let _bg_mad = mad_distance(&ring_samples, bg_median);
    let bg_uniform_argb = uniform_bg_argb(&ring_samples, bg_median);

    let mut luma_hist = [0u32; 256];
    let mut poly_count = 0usize;
    for idx in 0..polygon_local.len() {
        if !polygon_local[idx] {
            continue;
        }
        let lx = (idx % roi_w_us) as u32;
        let ly = (idx / roi_w_us) as u32;
        let gx = roi.left + lx;
        let gy = roi.top + ly;
        let p = image.get_pixel(gx, gy);
        luma_hist[luma(*p) as usize] += 1;
        poly_count += 1;
    }
    if poly_count < 16 {
        return None;
    }
    let otsu_threshold = otsu_split(&luma_hist);
    let ink_is_dark = decide_ink_class(
        image,
        roi,
        polygon_local.as_slice(),
        roi_w_us,
        otsu_threshold,
        bg_median,
    )?;

    let on_ink_side = |p: Rgba<u8>| -> bool {
        if ink_is_dark {
            luma(p) < otsu_threshold
        } else {
            luma(p) > otsu_threshold
        }
    };

    let mut seed = vec![false; polygon_local.len()];
    for idx in 0..polygon_local.len() {
        if !polygon_local[idx] {
            continue;
        }
        let lx = (idx % roi_w_us) as u32;
        let ly = (idx / roi_w_us) as u32;
        let p = *image.get_pixel(roi.left + lx, roi.top + ly);
        if on_ink_side(p) {
            seed[idx] = true;
        }
    }
    let outside_bias = 18i32;
    let outside_cut = if ink_is_dark {
        (otsu_threshold as i32 + outside_bias).min(255) as u8
    } else {
        (otsu_threshold as i32 - outside_bias).max(0) as u8
    };
    let on_ink_side_at = |p: Rgba<u8>, cut: u8| -> bool {
        if ink_is_dark {
            luma(p) < cut
        } else {
            luma(p) > cut
        }
    };
    let mut candidate = vec![false; polygon_local.len()];
    for idx in 0..polygon_local.len() {
        if !classify_region[idx] {
            continue;
        }
        let lx = (idx % roi_w_us) as u32;
        let ly = (idx / roi_w_us) as u32;
        let gx = roi.left + lx;
        let gy = roi.top + ly;
        if !cmask.contains(gx, gy) && contour_occupancy[(gy as usize) * (w as usize) + gx as usize]
        {
            continue;
        }
        let p = *image.get_pixel(gx, gy);
        let cut = if polygon_local[idx] {
            otsu_threshold
        } else {
            outside_cut
        };
        if on_ink_side_at(p, cut) {
            candidate[idx] = true;
        }
    }
    let ink_core = hysteresis_flood_with_cca(&seed, &candidate, roi_w, roi_h, line_h);

    let small_pad = ((line_h * 12 + 50) / 100).max(2);
    let base_mask = dilate(&polygon_local, roi_w, roi_h, small_pad);

    let extension_seed: Vec<bool> = ink_core
        .iter()
        .zip(polygon_local.iter())
        .map(|(&core, &poly)| core && !poly)
        .collect();
    let extension_cap_radius = (line_h / 4).clamp(2, 8);
    let extension_cap = dilate(&polygon_local, roi_w, roi_h, extension_cap_radius);
    let extension_dilated = dilate(&extension_seed, roi_w, roi_h, 1);
    let extension_capped: Vec<bool> = extension_dilated
        .iter()
        .zip(extension_cap.iter())
        .map(|(&e, &c)| e && c)
        .collect();

    let ink: Vec<bool> = base_mask
        .iter()
        .zip(extension_capped.iter())
        .map(|(&a, &b)| a || b)
        .collect();

    let mut poly_total = 0usize;
    let mut core_in_poly = 0usize;
    for idx in 0..polygon_local.len() {
        if polygon_local[idx] {
            poly_total += 1;
            if ink_core[idx] {
                core_in_poly += 1;
            }
        }
    }

    let mut ink_pixels = Vec::new();
    for ly in 0..roi_h {
        for lx in 0..roi_w {
            if ink[(ly as usize) * roi_w_us + lx as usize] {
                ink_pixels.push((roi.left + lx, roi.top + ly));
            }
        }
    }
    if ink_pixels.len() < 6 {
        return None;
    }

    let poly_coverage_pct = if poly_total > 0 {
        (core_in_poly * 100 / poly_total) as u32
    } else {
        0
    };
    Some(Detection {
        box_index: cmask.box_index,
        contour_rect: cmask.rect,
        bg_ring_median: bg_median,
        ink_pixels,
        fell_back: false,
        poly_coverage_pct,
        ink_is_dark,
        bg_uniform_argb,
    })
}

/// Decide whether the ring samples cluster around a single bg colour
/// (return `Some(argb)`) or spread across multiple — gradient, shadow
/// boundary, mixed bg — in which case the caller should fall back to
/// the neutral dark pill (`None`).
///
/// Approach: 4-bit-per-channel histogram (16³ = 4096 buckets) over the
/// ring's RGB. For each bucket, sum its own count plus the counts in
/// its 26 axis-neighbour buckets (3×3×3 cube minus the centre). This
/// "smoothed" count is robust to sensor noise that straddles a bucket
/// boundary — a uniform white wall with ±4 levels of noise won't get
/// artificially split across two buckets. If the smoothed peak holds
/// at least `UNIFORM_PEAK_THRESHOLD` of the total samples, the bg is
/// uniform; return the original median as the representative colour
/// (cheaper and just as accurate as a per-cluster mean given the
/// noise floor we're in).
fn uniform_bg_argb(samples: &[Rgba<u8>], median: Rgba<u8>) -> Option<u32> {
    /// Smoothed-peak fraction required to call the bg "uniform."
    /// 0.6 is a starting point: pure white walls land near 1.0,
    /// gentle shadows drop into the 0.3–0.5 range. Tune from device
    /// data once we see how it behaves in the wild.
    const UNIFORM_PEAK_THRESHOLD: f32 = 0.6;
    const SHIFT: u32 = 4;
    const LEVELS: u32 = 16;
    if samples.is_empty() {
        return None;
    }
    // 4096-entry sparse histogram. HashMap is enough at this size; we
    // run once per detection (not per frame) so the alloc is fine.
    let mut hist: std::collections::HashMap<(u8, u8, u8), u32> = std::collections::HashMap::new();
    for c in samples {
        let key = (c[0] >> SHIFT, c[1] >> SHIFT, c[2] >> SHIFT);
        *hist.entry(key).or_insert(0) += 1;
    }
    let total = samples.len() as f32;
    let mut best_smoothed: u32 = 0;
    for (&(r, g, b), _) in &hist {
        let mut s = 0u32;
        for dr in -1i32..=1 {
            let rr = r as i32 + dr;
            if rr < 0 || rr >= LEVELS as i32 {
                continue;
            }
            for dg in -1i32..=1 {
                let gg = g as i32 + dg;
                if gg < 0 || gg >= LEVELS as i32 {
                    continue;
                }
                for db in -1i32..=1 {
                    let bb = b as i32 + db;
                    if bb < 0 || bb >= LEVELS as i32 {
                        continue;
                    }
                    s += hist
                        .get(&(rr as u8, gg as u8, bb as u8))
                        .copied()
                        .unwrap_or(0);
                }
            }
        }
        if s > best_smoothed {
            best_smoothed = s;
        }
    }
    if (best_smoothed as f32) / total >= UNIFORM_PEAK_THRESHOLD {
        Some(rgba_to_argb(median))
    } else {
        None
    }
}

fn rgba_to_argb(c: Rgba<u8>) -> u32 {
    let a = c[3] as u32;
    let r = c[0] as u32;
    let g = c[1] as u32;
    let b = c[2] as u32;
    (a << 24) | (r << 16) | (g << 8) | b
}

/// Estimate a per-detection ink colour by taking the median of ink
/// pixels in the top quartile of bg-distance. Robust to anti-aliased
/// rim pixels (which sit close to the bg in colour space). Currently
/// unused by `mat_detections` itself — the smoke test still calls it
/// for diagnostics, and future "tint the inpainted strip with original
/// ink colour" experiments would reuse it.
pub(crate) fn estimate_fg(
    image: &RgbaImage,
    inpainted: &RgbaImage,
    ink_pixels: &[(u32, u32)],
) -> Rgba<u8> {
    if ink_pixels.is_empty() {
        return Rgba([0, 0, 0, 255]);
    }
    let mut scored: Vec<(f32, Rgba<u8>)> = ink_pixels
        .iter()
        .map(|&(px, py)| {
            let original = *image.get_pixel(px, py);
            let bg = *inpainted.get_pixel(px, py);
            (rgb_dist2(original, bg), original)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = (scored.len() / 4).max(1);
    let top: Vec<Rgba<u8>> = scored.into_iter().take(take).map(|(_, c)| c).collect();
    median_color(&top)
}

/// 4-direction nearest-non-ink inpaint with median outlier rejection
/// and post-pass smoothing. Sees the same `(image, mask)` shape as the
/// rest of the matting code so it can be reused on dewarped strips
/// (the path `mat_strip_for_detection` takes) or, in principle, on
/// canonical-frame coords directly (the legacy path the smoke test
/// also tries).
///
/// - `max_search`: cut sample propagation off after this many pixels
///   along an axis. Beyond this, the direction reports "no sample"
///   so the median is computed from the remaining directions only.
/// - `sample_radius`: dilate the mask before walking so rim
///   anti-aliased pixels don't get selected as bg samples.
/// - `smoothing_passes`: N×3 box blur over masked pixels only. Kills
///   axis-aligned banding from the 4-direction walk.
pub(crate) fn inpaint_native(
    image: &[Rgba<u8>],
    mask: &[bool],
    w: u32,
    h: u32,
    max_search: u32,
    sample_radius: u32,
    smoothing_passes: u32,
) -> Vec<Rgba<u8>> {
    let w_us = w as usize;
    let h_us = h as usize;
    let n = w_us * h_us;
    let sample_mask = dilate(mask, w, h, sample_radius);
    let mask_for_walk = sample_mask.as_slice();

    let mut up = vec![(u32::MAX, Rgba([0u8; 4])); n];
    let mut down = vec![(u32::MAX, Rgba([0u8; 4])); n];
    let mut left = vec![(u32::MAX, Rgba([0u8; 4])); n];
    let mut right = vec![(u32::MAX, Rgba([0u8; 4])); n];

    let mut row_has_mask = vec![false; h_us];
    for y in 0..h_us {
        let row = y * w_us;
        for x in 0..w_us {
            if mask_for_walk[row + x] {
                row_has_mask[y] = true;
                break;
            }
        }
    }

    for x in 0..w {
        let mut state: (u32, Rgba<u8>) = (u32::MAX, Rgba([0; 4]));
        for y in 0..h {
            let idx = (y as usize) * w_us + x as usize;
            if mask_for_walk[idx] {
                if state.0 != u32::MAX {
                    state.0 = state.0.saturating_add(1);
                    if state.0 > max_search {
                        state.0 = u32::MAX;
                    }
                }
            } else {
                state = (0, image[idx]);
            }
            up[idx] = state;
        }
        state = (u32::MAX, Rgba([0; 4]));
        for y in (0..h).rev() {
            let idx = (y as usize) * w_us + x as usize;
            if mask_for_walk[idx] {
                if state.0 != u32::MAX {
                    state.0 = state.0.saturating_add(1);
                    if state.0 > max_search {
                        state.0 = u32::MAX;
                    }
                }
            } else {
                state = (0, image[idx]);
            }
            down[idx] = state;
        }
    }
    for y in 0..h_us {
        if !row_has_mask[y] {
            continue;
        }
        let mut state: (u32, Rgba<u8>) = (u32::MAX, Rgba([0; 4]));
        for x in 0..w_us {
            let idx = y * w_us + x;
            if mask_for_walk[idx] {
                if state.0 != u32::MAX {
                    state.0 = state.0.saturating_add(1);
                    if state.0 > max_search {
                        state.0 = u32::MAX;
                    }
                }
            } else {
                state = (0, image[idx]);
            }
            left[idx] = state;
        }
        state = (u32::MAX, Rgba([0; 4]));
        for x in (0..w_us).rev() {
            let idx = y * w_us + x;
            if mask_for_walk[idx] {
                if state.0 != u32::MAX {
                    state.0 = state.0.saturating_add(1);
                    if state.0 > max_search {
                        state.0 = u32::MAX;
                    }
                }
            } else {
                state = (0, image[idx]);
            }
            right[idx] = state;
        }
    }

    let mut out = image.to_vec();
    let outlier_thr = 35.0f32 * 35.0;
    for y in 0..h_us {
        if !row_has_mask[y] {
            continue;
        }
        for x in 0..w_us {
            let idx = y * w_us + x;
            if !mask[idx] {
                continue;
            }
            let raw = [up[idx], down[idx], left[idx], right[idx]];
            let mut count = 0usize;
            let mut rs = [0u8; 4];
            let mut gs = [0u8; 4];
            let mut bs = [0u8; 4];
            for &(d, c) in &raw {
                if d == u32::MAX {
                    continue;
                }
                rs[count] = c[0];
                gs[count] = c[1];
                bs[count] = c[2];
                count += 1;
            }
            if count == 0 {
                continue;
            }
            rs[..count].sort_unstable();
            gs[..count].sort_unstable();
            bs[..count].sort_unstable();
            let median = Rgba([rs[count / 2], gs[count / 2], bs[count / 2], 255]);

            let mut wsum = 0.0f32;
            let mut rsum = 0.0f32;
            let mut gsum = 0.0f32;
            let mut bsum = 0.0f32;
            for &(d, c) in &raw {
                if d == u32::MAX {
                    continue;
                }
                if rgb_dist2(c, median) > outlier_thr {
                    continue;
                }
                let weight = 1.0 / (d as f32 + 1.0);
                wsum += weight;
                rsum += weight * c[0] as f32;
                gsum += weight * c[1] as f32;
                bsum += weight * c[2] as f32;
            }
            out[idx] = if wsum == 0.0 {
                median
            } else {
                Rgba([
                    (rsum / wsum).round().clamp(0.0, 255.0) as u8,
                    (gsum / wsum).round().clamp(0.0, 255.0) as u8,
                    (bsum / wsum).round().clamp(0.0, 255.0) as u8,
                    255,
                ])
            };
        }
    }

    let mut updates: Vec<(usize, Rgba<u8>)> = Vec::new();
    for _ in 0..smoothing_passes {
        updates.clear();
        for y in 0..h_us {
            if !row_has_mask[y] {
                continue;
            }
            for x in 0..w_us {
                let idx = y * w_us + x;
                if !mask[idx] {
                    continue;
                }
                let x0 = x.saturating_sub(1);
                let y0 = y.saturating_sub(1);
                let x1 = (x + 1).min(w_us - 1);
                let y1 = (y + 1).min(h_us - 1);
                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                let mut cnt = 0u32;
                for yy in y0..=y1 {
                    let row = yy * w_us;
                    for xx in x0..=x1 {
                        let p = out[row + xx];
                        r += p[0] as u32;
                        g += p[1] as u32;
                        b += p[2] as u32;
                        cnt += 1;
                    }
                }
                updates.push((
                    idx,
                    Rgba([(r / cnt) as u8, (g / cnt) as u8, (b / cnt) as u8, 255]),
                ));
            }
        }
        for (idx, p) in updates.iter() {
            out[*idx] = *p;
        }
    }
    out
}

pub(crate) fn hysteresis_flood_with_cca(
    seed: &[bool],
    candidate: &[bool],
    w: u32,
    h: u32,
    line_h: u32,
) -> Vec<bool> {
    let w_us = w as usize;
    let mut out = vec![false; seed.len()];
    let mut stack: Vec<(u32, u32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y as usize) * w_us + x as usize;
            if seed[idx] {
                out[idx] = true;
                stack.push((x, y));
            }
        }
    }
    while let Some((cx, cy)) = stack.pop() {
        let x0 = cx.saturating_sub(1);
        let y0 = cy.saturating_sub(1);
        let x1 = (cx + 1).min(w - 1);
        let y1 = (cy + 1).min(h - 1);
        for ny in y0..=y1 {
            for nx in x0..=x1 {
                let nidx = (ny as usize) * w_us + nx as usize;
                if out[nidx] || !candidate[nidx] {
                    continue;
                }
                out[nidx] = true;
                stack.push((nx, ny));
            }
        }
    }

    let max_dim = (line_h as f32 * 1.2) as u32 + 2;
    let min_pixels = 3usize;
    let max_pixels = (line_h * line_h) as usize;
    let mut visited = out.clone();
    for sy in 0..h {
        for sx in 0..w {
            let sidx = (sy as usize) * w_us + sx as usize;
            if visited[sidx] || !candidate[sidx] {
                continue;
            }
            let mut comp: Vec<usize> = Vec::new();
            let mut min_x = sx;
            let mut max_x = sx;
            let mut min_y = sy;
            let mut max_y = sy;
            visited[sidx] = true;
            stack.push((sx, sy));
            while let Some((cx, cy)) = stack.pop() {
                let cidx = (cy as usize) * w_us + cx as usize;
                comp.push(cidx);
                if cx < min_x {
                    min_x = cx;
                }
                if cx > max_x {
                    max_x = cx;
                }
                if cy < min_y {
                    min_y = cy;
                }
                if cy > max_y {
                    max_y = cy;
                }
                let x0 = cx.saturating_sub(1);
                let y0 = cy.saturating_sub(1);
                let x1 = (cx + 1).min(w - 1);
                let y1 = (cy + 1).min(h - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let nidx = (ny as usize) * w_us + nx as usize;
                        if visited[nidx] || !candidate[nidx] {
                            continue;
                        }
                        visited[nidx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            let bbox_w = max_x - min_x + 1;
            let bbox_h = max_y - min_y + 1;
            if comp.len() < min_pixels
                || comp.len() > max_pixels
                || bbox_h > max_dim
                || bbox_w > max_dim
            {
                continue;
            }
            for cidx in comp {
                out[cidx] = true;
            }
        }
    }
    out
}

pub(crate) fn dilate(mask: &[bool], w: u32, h: u32, radius: u32) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let w_us = w as usize;
    let h_us = h as usize;
    let r = radius as usize;

    let mut tmp = vec![false; mask.len()];
    for y in 0..h_us {
        let row = y * w_us;
        let mut count: u32 = 0;
        for x in 0..(r + 1).min(w_us) {
            if mask[row + x] {
                count += 1;
            }
        }
        for x in 0..w_us {
            tmp[row + x] = count > 0;
            let add_x = x + r + 1;
            if add_x < w_us && mask[row + add_x] {
                count += 1;
            }
            if x >= r && mask[row + (x - r)] {
                count -= 1;
            }
        }
    }
    let mut out = vec![false; mask.len()];
    for x in 0..w_us {
        let mut count: u32 = 0;
        for y in 0..(r + 1).min(h_us) {
            if tmp[y * w_us + x] {
                count += 1;
            }
        }
        for y in 0..h_us {
            out[y * w_us + x] = count > 0;
            let add_y = y + r + 1;
            if add_y < h_us && tmp[add_y * w_us + x] {
                count += 1;
            }
            if y >= r && tmp[(y - r) * w_us + x] {
                count -= 1;
            }
        }
    }
    out
}

pub(crate) fn median_color(colors: &[Rgba<u8>]) -> Rgba<u8> {
    if colors.is_empty() {
        return Rgba([0, 0, 0, 255]);
    }
    let mut r: Vec<u8> = colors.iter().map(|c| c[0]).collect();
    let mut g: Vec<u8> = colors.iter().map(|c| c[1]).collect();
    let mut b: Vec<u8> = colors.iter().map(|c| c[2]).collect();
    r.sort_unstable();
    g.sort_unstable();
    b.sort_unstable();
    let mid = r.len() / 2;
    Rgba([r[mid], g[mid], b[mid], 255])
}

pub(crate) fn mad_distance(samples: &[Rgba<u8>], median: Rgba<u8>) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut dists: Vec<f32> = samples
        .iter()
        .map(|c| rgb_dist2(*c, median).sqrt())
        .collect();
    dists.sort_unstable_by(f32::total_cmp);
    dists[dists.len() / 2]
}

pub(crate) fn decide_ink_class(
    image: &RgbaImage,
    roi: Rect,
    polygon_local: &[bool],
    roi_w_us: usize,
    otsu_threshold: u8,
    bg_median: Rgba<u8>,
) -> Option<bool> {
    let mut dark_colors = Vec::new();
    let mut light_colors = Vec::new();
    for idx in 0..polygon_local.len() {
        if !polygon_local[idx] {
            continue;
        }
        let lx = (idx % roi_w_us) as u32;
        let ly = (idx / roi_w_us) as u32;
        let p = *image.get_pixel(roi.left + lx, roi.top + ly);
        if luma(p) < otsu_threshold {
            dark_colors.push(p);
        } else {
            light_colors.push(p);
        }
    }
    if dark_colors.is_empty() || light_colors.is_empty() {
        return None;
    }
    let dark_median = median_color(&dark_colors);
    let light_median = median_color(&light_colors);
    Some(rgb_dist2(dark_median, bg_median) >= rgb_dist2(light_median, bg_median))
}

pub(crate) fn luma(c: Rgba<u8>) -> u8 {
    let r = c[0] as u32;
    let g = c[1] as u32;
    let b = c[2] as u32;
    ((299 * r + 587 * g + 114 * b) / 1000).min(255) as u8
}

pub(crate) fn otsu_split(hist: &[u32; 256]) -> u8 {
    let total: u64 = hist.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return 128;
    }
    let mut sum_total: u64 = 0;
    for i in 0..256 {
        sum_total += (i as u64) * (hist[i] as u64);
    }
    let mut sum_bg: u64 = 0;
    let mut w_bg: u64 = 0;
    let mut best_var = -1.0f64;
    let mut best_t: u8 = 128;
    for t in 0..256 {
        w_bg += hist[t] as u64;
        if w_bg == 0 {
            continue;
        }
        let w_fg = total - w_bg;
        if w_fg == 0 {
            break;
        }
        sum_bg += (t as u64) * (hist[t] as u64);
        let mean_bg = sum_bg as f64 / w_bg as f64;
        let mean_fg = (sum_total - sum_bg) as f64 / w_fg as f64;
        let diff = mean_bg - mean_fg;
        let var = w_bg as f64 * w_fg as f64 * diff * diff;
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

pub(crate) fn rgb_dist2(a: Rgba<u8>, b: Rgba<u8>) -> f32 {
    let dr = a[0] as f32 - b[0] as f32;
    let dg = a[1] as f32 - b[1] as f32;
    let db = a[2] as f32 - b[2] as f32;
    dr * dr + dg * dg + db * db
}

pub(crate) fn clamp_rect(rect: Rect, w: u32, h: u32) -> Option<Rect> {
    let out = Rect {
        left: rect.left.min(w),
        top: rect.top.min(h),
        right: rect.right.min(w),
        bottom: rect.bottom.min(h),
    };
    if out.right > out.left && out.bottom > out.top {
        Some(out)
    } else {
        None
    }
}

pub(crate) fn inflate_rect_xy(rect: Rect, pad_x: u32, pad_y: u32, w: u32, h: u32) -> Rect {
    Rect {
        left: rect.left.saturating_sub(pad_x),
        top: rect.top.saturating_sub(pad_y),
        right: rect.right.saturating_add(pad_x).min(w),
        bottom: rect.bottom.saturating_add(pad_y).min(h),
    }
}

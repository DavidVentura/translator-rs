//! Per-detection color matting: take the ink model's per-box matte for
//! each recognised text region and produce an inpainted rectified strip
//! suitable for use as a live-overlay background. Replaces the hardcoded
//! dark "pill" with a real per-pixel reconstruction of the camera
//! background under the source text.
//!
//! ## Why a rectified strip
//!
//! The inpaint operates in **rectified strip coordinates**, where the
//! text runs strictly left-to-right inside an axis-aligned bitmap.
//! Column-wise inpaint assumes glyphs are bounded vertically per
//! column with a contiguous masked region between non-ink pixels — an
//! assumption that breaks on any tilted text. Dewarping into rectified
//! coords first restores the assumption.
//!
//! At acquire time, the `MattedStrip` produced here feeds the GPU overlay
//! compositor: its color drives the pill, and the GPU warps the baked overlay
//! per-frame by the planar-tracker homography (canonical → viewport coords).
//!
//! ## Pipeline per detection
//!
//! 1. Take the ink model's soft 0..255 matte for the box (from
//!    `PpocrEngine::ink_masks`, rendered in the box's oriented-box frame).
//! 2. Project every box's matte into one image-space *union* ink mask:
//!    walk the box's source-space bounding box, map each pixel back to
//!    the matte via the oriented frame, and set the union where the alpha
//!    clears `INK_ALPHA_CUT`. The union lets a tall strip erase the
//!    neighbouring lines that fall inside its padding.
//! 3. Derive the box's ink class (dark/light) and a uniform-bg colour
//!    from the pixels the matte partitions — the model says *what* is
//!    ink, the pixels say what colour it and the background are.
//! 4. Dewarp the camera + union ink mask into a rectified strip aligned
//!    to the oriented box, growing the fill mask by a height-proportional
//!    radius so the original ink's anti-aliased rim is erased too.
//! 5. Inpaint masked pixels in the strip via 4-direction nearest-non-ink
//!    blend with outlier rejection + smoothing passes.
//!
//! Output: rectified RGBA strip + the oriented-box parameters
//! needed to re-warp it back to canonical-frame coords.

use image::{GrayImage, Rgba, RgbaImage};

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

/// Confidence above which a model ink-mask pixel is treated as ink (to
/// erase). The model emits a soft 0..255 alpha; the inpaint then dilates
/// by `inpaint_sample_radius` to catch the anti-aliased rim, so a
/// mid-grey cut is enough without bleeding into the background.
const INK_ALPHA_CUT: u8 = 40;
/// Minimum ink pixels in a strip to bother matting it. Below this the
/// model found essentially no ink in the box — return `None` and let the
/// caller fall back to default-pill rendering.
const MIN_INK_PIXELS: usize = 6;
/// Minimum background samples needed before judging the bg "uniform".
const MIN_BG_SAMPLES: usize = 24;

/// Per-box ink metadata the model mask can't give us directly: the ink
/// class (for picking a readable translated-text colour) and a
/// uniform-background colour (for the solid-pill fallback). Both are
/// derived from the box pixels the mask partitions into ink/bg.
#[derive(Clone, Copy)]
struct BoxInk {
    ink_is_dark: bool,
    bg_uniform_argb: Option<u32>,
}

/// Compute matted strips for each detection from the ink model's per-box
/// masks. `ink_masks` is 1:1 with `boxes` (the output of
/// `PpocrEngine::ink_masks`); each mask is a soft 0..255 alpha in the
/// box's oriented-box rectified space.
///
/// The masks are first projected into a single image-space *union* ink
/// mask. A strip is padded ~75% of the line height beyond the text band,
/// so a tall strip overlaps neighbouring lines; sampling the union (not
/// just this box's mask) erases those neighbours' glyphs inside the strip
/// too. Without it, a later strip would composite a neighbour's untouched
/// original text back over a line an earlier strip had already erased.
///
/// Returns one entry per input box; entries are `None` when the box has
/// no model mask, a degenerate oriented box, or the model found no ink —
/// the caller falls back to default-pill rendering for those.
pub fn mat_detections(
    rgba: &RgbaImage,
    boxes: &[DetectedTextBox],
    ink_masks: &[Option<GrayImage>],
) -> Vec<Option<MattedStrip>> {
    mat_detections_with_config(rgba, boxes, ink_masks, &MattingConfig::default())
}

/// Like [`mat_detections`] but with explicit tuning. Mostly useful for
/// tests/benches.
pub fn mat_detections_with_config(
    rgba: &RgbaImage,
    boxes: &[DetectedTextBox],
    ink_masks: &[Option<GrayImage>],
    cfg: &MattingConfig,
) -> Vec<Option<MattedStrip>> {
    let (w, h) = rgba.dimensions();
    let mut union_ink = vec![false; (w as usize) * (h as usize)];
    let mut meta: Vec<Option<BoxInk>> = Vec::with_capacity(boxes.len());
    meta.resize_with(boxes.len(), || None);
    for (idx, b) in boxes.iter().enumerate() {
        let Some(Some(mask)) = ink_masks.get(idx) else {
            continue;
        };
        meta[idx] = project_box_ink(rgba, b, mask, w, h, &mut union_ink);
    }

    boxes
        .iter()
        .enumerate()
        .map(|(idx, b)| {
            let ink = meta[idx]?;
            mat_strip_for_detection(rgba, idx, b, &union_ink, w, h, ink, cfg)
        })
        .collect()
}

/// Project one box's model mask into the shared image-space `union_ink`
/// (set a source pixel when its `(u, v)` in the box's oriented frame maps
/// to a model-mask alpha above the cut), and derive the box's ink class
/// and uniform-bg colour from the pixels the mask partitions. Iterates
/// the box's source-space bounding box directly, so the union is dense at
/// full resolution with no projection gaps. Returns `None` when the model
/// found essentially no ink.
fn project_box_ink(
    image: &RgbaImage,
    detected: &DetectedTextBox,
    ink_mask: &GrayImage,
    w: u32,
    h: u32,
    union_ink: &mut [bool],
) -> Option<BoxInk> {
    let o = detected.oriented_box;
    if o.width <= 1.0 || o.height <= 1.0 {
        return None;
    }
    let cos_a = o.angle_radians.cos();
    let sin_a = o.angle_radians.sin();
    let half_w = o.width * 0.5;
    let half_h = o.height * 0.5;
    let mw = ink_mask.width() as f32;
    let mh = ink_mask.height() as f32;

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for (u, v) in [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ] {
        let px = u * cos_a - v * sin_a + o.cx;
        let py = u * sin_a + v * cos_a + o.cy;
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    }
    let x0 = min_x.floor().max(0.0) as u32;
    let y0 = min_y.floor().max(0.0) as u32;
    let x1 = (max_x.ceil().max(0.0) as u32).min(w);
    let y1 = (max_y.ceil().max(0.0) as u32).min(h);

    let w_us = w as usize;
    let mut ink_samples: Vec<Rgba<u8>> = Vec::new();
    let mut bg_samples: Vec<Rgba<u8>> = Vec::new();
    for py in y0..y1 {
        for px in x0..x1 {
            let dx = px as f32 + 0.5 - o.cx;
            let dy = py as f32 + 0.5 - o.cy;
            let u = dx * cos_a + dy * sin_a;
            let v = -dx * sin_a + dy * cos_a;
            if u.abs() > half_w || v.abs() > half_h {
                continue;
            }
            let mx = (((u + half_w) / o.width) * mw).floor().clamp(0.0, mw - 1.0) as u32;
            let my = (((v + half_h) / o.height) * mh)
                .floor()
                .clamp(0.0, mh - 1.0) as u32;
            let pixel = *image.get_pixel(px, py);
            if ink_mask.get_pixel(mx, my)[0] >= INK_ALPHA_CUT {
                union_ink[(py as usize) * w_us + px as usize] = true;
                ink_samples.push(pixel);
            } else {
                bg_samples.push(pixel);
            }
        }
    }

    if ink_samples.len() < MIN_INK_PIXELS {
        return None;
    }
    let ink_is_dark = if bg_samples.is_empty() {
        true
    } else {
        luma(median_color(&ink_samples)) < luma(median_color(&bg_samples))
    };
    let bg_uniform_argb = if bg_samples.len() >= MIN_BG_SAMPLES {
        uniform_bg_argb(&bg_samples, median_color(&bg_samples))
    } else {
        None
    };
    Some(BoxInk {
        ink_is_dark,
        bg_uniform_argb,
    })
}

/// Dewarp a detection's oriented box into a rectified RGBA strip, sample
/// the shared `union_ink` into strip coords, then inpaint the masked
/// pixels. The strip is padded vertically by ~75% of the line height so
/// the inpaint walk has clean background above and below the text band;
/// horizontally by 15% for translated-text breathing room.
fn mat_strip_for_detection(
    image: &RgbaImage,
    box_index: usize,
    detected: &DetectedTextBox,
    union_ink: &[bool],
    w: u32,
    h: u32,
    ink: BoxInk,
    cfg: &MattingConfig,
) -> Option<MattedStrip> {
    let oriented = detected.oriented_box;
    if oriented.width <= 1.0 || oriented.height <= 1.0 {
        return None;
    }
    let cos_a = oriented.angle_radians.cos();
    let sin_a = oriented.angle_radians.sin();

    let pad_x = (oriented.width * 0.15).max(4.0);
    let pad_y = (oriented.height * 0.75).max(8.0);
    let strip_w = (oriented.width + 2.0 * pad_x).ceil().max(8.0) as u32;
    let strip_h = (oriented.height + 2.0 * pad_y).ceil().max(8.0) as u32;
    let sw_us = strip_w as usize;
    let w_us = w as usize;
    let strip_cx = strip_w as f32 * 0.5;
    let strip_cy = strip_h as f32 * 0.5;

    // Sample the image + union ink mask into strip coords via inverse
    // warp. Out-of-image pixels are flagged masked so the inpaint walk
    // skips them.
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
            strip_mask[idx] = union_ink[(pyi as usize) * w_us + pxi as usize];
        }
    }

    // The model mask is glyph-tight and sampled at 48px, so on large
    // glyphs its upscaled edge sits inside the original ink's anti-aliased
    // rim. Grow the *fill* region by a height-proportional radius so that
    // rim is inpainted too, instead of surviving as a faint outline.
    let fill_radius = ((oriented.height * 0.06).round() as u32).clamp(1, 6);
    let strip_mask = dilate(&strip_mask, strip_w, strip_h, fill_radius);
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
        box_index,
        strip_rgba: strip_bytes,
        strip_width: strip_w,
        strip_height: strip_h,
        canonical_cx: oriented.cx,
        canonical_cy: oriented.cy,
        canonical_angle_radians: oriented.angle_radians,
        canonical_width: strip_w as f32,
        canonical_height: strip_h as f32,
        ink_is_dark: ink.ink_is_dark,
        bg_uniform_argb: ink.bg_uniform_argb,
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

pub(crate) fn luma(c: Rgba<u8>) -> u8 {
    let r = c[0] as u32;
    let g = c[1] as u32;
    let b = c[2] as u32;
    ((299 * r + 587 * g + 114 * b) / 1000).min(255) as u8
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

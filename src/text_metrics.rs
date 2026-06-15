//! Per-line text metrics measured from the ink model's per-line matte.
//!
//! The detection box gives a coarse oriented rect; its height mixes ascenders,
//! descenders and caps, so it varies with *which glyphs* a line happens to
//! contain, its angle (fitted on the raw contour) skews when a tall letter drags
//! the principal axis, and its width clips ~half a glyph off each end. The ink
//! model already runs per line (it also drives colour matting, for free), and its
//! matte sees only ink — so it's the right place to recover the line's true
//! typography:
//!
//! - **x-height** — the dense, column-coherent central band, stable across glyph
//!   content because it ignores the sparse ascender/descender rows.
//! - **centreline & width** — the ink's actual centre and full horizontal extent,
//!   not the (off-centre, clipped) detection box.
//! - **baseline tilt** — the residual lean the rough oriented-box angle missed,
//!   measured from the band's per-column centroid, immune to the ascender drag
//!   that fools contour PCA.
//!
//! These feed paragraph grouping (a glyph-stable size/position beats the wobbly
//! detection box). The same matte can later yield more metrics — stroke width for
//! bold detection, slant for italics — so this is the line's text-metric layer,
//! not a matting detail.
//!
//! Everything here is computed in the matte's rectified strip space (the ink
//! model's per-line output: `mw` columns spanning the box width along the reading
//! axis, `mh` rows — 48 — spanning the box height across it). A stray background
//! blob the model occasionally grabs, or bleed from a neighbouring line through
//! the box's vertical padding, shows up as a band *off* the strip centre; the
//! central-band pick drops it without per-pixel classification.

use image::GrayImage;

use crate::ocr::OrientedRect;

/// Matte alpha at or above which a texel counts as ink. Matches the erase
/// path's `INK_ALPHA_CUT` so band detection and erasure agree on what is ink.
const INK_CUT: u16 = 40;
/// Fraction of the peak row-profile that marks a row as part of the line's
/// vertical *support* (as opposed to inter-line gap or padding). Deliberately
/// low: it only has to separate this line's ink from a neighbouring line that
/// bled into the box's padding (which shows as a *disjoint* run), not to find
/// the x-height edge — that comes from the ink-mass percentiles below.
const SUPPORT_FRAC: f32 = 0.12;
/// Ink-mass percentiles bounding the x-height band. Walking the cumulative row
/// profile and cutting at these fractions of total ink is robust to a single
/// dense row (e.g. the x-line where every lowercase top piles up) and to a
/// shallow mid-band dip — both of which fragment a fraction-of-peak threshold.
const MASS_LO: f32 = 0.10;
const MASS_HI: f32 = 0.90;
/// Minimum inked columns to attempt a baseline-tilt fit. Below this the slope
/// is noise; report zero tilt and keep just the x-height.
const MIN_TILT_COLUMNS: usize = 8;
/// Clamp on the recovered tilt. The matte was produced from a strip dewarped
/// with the rough angle, so a real residual is small; a larger fit is a
/// degenerate matte, not a steeper line.
const MAX_TILT_RADIANS: f32 = 0.26; // ~15°
/// Stroke width is measured on the confident *core* of the ink — texels at or
/// above this fraction of the line's own peak alpha. Relative (not a fixed cut)
/// so a faint/low-confidence bold matte isn't penalised the way a fixed high
/// threshold would; feathered edges, which matter proportionally less on a bold
/// stroke, drop out either way. Floored at `INK_CUT` so a weak matte still has a
/// core.
const STROKE_CORE_FRAC: f32 = 0.6;
/// Stroke-core-width-to-x-height ratio at or above which a line reads as bold.
/// Calibrated on real photographed print pages: regular body runs ~0.15–0.20,
/// bold headings ~0.27+. Note all-caps lines read low (their band is cap-height,
/// not x-height), so a large all-caps title can slip under this.
pub const BOLD_WEIGHT_RATIO: f32 = 0.26;

/// Typography recovered from one line's ink matte, in the source image's pixel
/// space (converted out of strip space via the box dimensions).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineMetrics {
    /// Height of the dense central ink band — an x-height-like metric stable
    /// across glyph content. Image-space pixels.
    pub x_height: f32,
    /// Signed offset of the band centre from the box centre, along the box's
    /// cross-reading (`v`) axis, in image-space pixels. Positive is toward +v
    /// (downward in strip space). Lets the caller pin the line's centreline to
    /// the actual ink rather than the (possibly off-centre) detection box.
    pub centerline_offset: f32,
    /// Full horizontal ink extent along the reading (`u`) axis, image-space
    /// pixels. The detection tight box undershoots the first/last glyph by
    /// ~half a glyph; the matte (computed on the wider inflated box) sees the
    /// whole line, so its column span recovers the true width.
    pub width: f32,
    /// Signed offset of the ink extent's centre from the box centre, along the
    /// reading (`u`) axis, image-space pixels. Re-centres the line on its actual
    /// ink rather than the detection box's (off-centre, clipped) span.
    pub center_u_offset: f32,
    /// Residual baseline tilt to add to the oriented box's angle (radians).
    /// Positive rotates the reading axis toward +v (downward in strip space).
    pub baseline_angle_delta: f32,
    /// Mean ink stroke width (image-space pixels), estimated as `2·area/perimeter`
    /// over the line's ink. On its own it scales with font size; divide by
    /// `x_height` ([`weight_ratio`]) for a size-invariant weight that separates
    /// bold from regular.
    pub stroke_width: f32,
}

impl LineMetrics {
    /// Stroke width relative to x-height — a size-invariant font-weight proxy.
    /// Regular body text sits well below [`BOLD_WEIGHT_RATIO`]; bold above it.
    pub fn weight_ratio(&self) -> f32 {
        self.stroke_width / self.x_height.max(1e-3)
    }

    /// Whether this line reads as bold, by its stroke-to-x-height ratio.
    pub fn is_bold(&self) -> bool {
        self.weight_ratio() >= BOLD_WEIGHT_RATIO
    }

    /// Re-fit a detection box to the measured ink: x-height as the height, the
    /// ink column span as the width, the centre snapped to the ink along both the
    /// reading (`u`) and cross-reading (`v`) axes, and the baseline tilt folded
    /// into the angle. Shared by the still pipeline, the integration test, and the
    /// `viz_pipeline` overlay so all three agree on what the matte says.
    pub fn refit(&self, base: OrientedRect) -> OrientedRect {
        let angle = base.angle_radians + self.baseline_angle_delta;
        let (sin, cos) = angle.sin_cos();
        OrientedRect {
            cx: base.cx + self.center_u_offset * cos - self.centerline_offset * sin,
            cy: base.cy + self.center_u_offset * sin + self.centerline_offset * cos,
            width: self.width,
            height: self.x_height,
            angle_radians: angle,
        }
    }
}

/// Recover [`LineMetrics`] from a box's ink matte. `box_width`/`box_height`
/// are the oriented box the matte was dewarped from, used to map strip rows back
/// to image-space pixels. Returns `None` when the matte holds no coherent band
/// (essentially no ink) — the caller keeps the box's own height in that case.
pub fn measure_line(matte: &GrayImage, box_width: f32, box_height: f32) -> Option<LineMetrics> {
    let (mw, mh) = matte.dimensions();
    if mw == 0 || mh == 0 || box_width <= 0.0 || box_height <= 0.0 {
        return None;
    }
    let data = matte.as_raw();
    let mw_us = mw as usize;

    let row_sum: Vec<u32> = (0..mh as usize)
        .map(|y| {
            let row = &data[y * mw_us..(y + 1) * mw_us];
            row.iter().map(|&a| a as u32).sum()
        })
        .collect();
    let peak = *row_sum.iter().max().unwrap();
    if peak == 0 {
        return None;
    }

    // Isolate the line's vertical support (drops neighbouring-line bleed, which
    // sits in a disjoint run off-centre), then read the x-height band from the
    // ink-mass distribution *within* that support.
    let (sup_top, sup_bottom) = central_band(&row_sum, peak as f32 * SUPPORT_FRAC, mh as usize)?;
    let total: f64 = row_sum[sup_top..=sup_bottom]
        .iter()
        .map(|&s| s as f64)
        .sum();
    let lo = mass_row(&row_sum, sup_top, sup_bottom, total, MASS_LO);
    let hi = mass_row(&row_sum, sup_top, sup_bottom, total, MASS_HI);
    let mid = mass_row(&row_sum, sup_top, sup_bottom, total, 0.5);

    let rows_to_px = box_height / mh as f32;
    let x_height = (hi - lo) * rows_to_px;
    let centerline_offset = (mid - (mh as f32 - 1.0) * 0.5) * rows_to_px;

    // Horizontal ink extent: the full span of inked columns within the support
    // band — the *full* span, not a percentile, since trimming would re-clip the
    // first/last glyph we're trying to recover.
    let (col_left, col_right) = ink_column_span(data, mw_us, sup_top, sup_bottom);
    let cols_to_px = box_width / mw as f32;
    let width = (col_right - col_left + 1) as f32 * cols_to_px;
    let ink_center_col = (col_left + col_right + 1) as f32 * 0.5;
    let center_u_offset = (ink_center_col - mw as f32 * 0.5) * cols_to_px;

    let slope_rows_per_col = baseline_slope(
        data,
        mw_us,
        mh as usize,
        lo.floor() as usize,
        hi.ceil() as usize,
    );
    // Strip column step → image displacement along the reading axis; row step →
    // displacement across it. The tilt is the angle between the band's traced
    // direction and the strip's column axis.
    let du = box_width / mw as f32;
    let dv = slope_rows_per_col * box_height / mh as f32;
    let baseline_angle_delta = dv.atan2(du).clamp(-MAX_TILT_RADIANS, MAX_TILT_RADIANS);

    let stroke_width = stroke_width_px(
        data,
        mw_us,
        mh as usize,
        sup_top,
        sup_bottom,
        rows_to_px,
        cols_to_px,
    );

    Some(LineMetrics {
        x_height,
        centerline_offset,
        width,
        center_u_offset,
        baseline_angle_delta,
        stroke_width,
    })
}

/// Mean stroke width over the line's ink, in image-space pixels, as
/// `2·area / perimeter`: for a stroke of width `w` and total length `L`,
/// `area ≈ w·L` and `perimeter ≈ 2·L`, so the ratio recovers `w`, averaged over
/// the whole line so junctions and serifs wash out. Perimeter counts ink texels
/// with a 4-neighbour that is non-ink (peeking the real strip rows above/below
/// the band so the band edge itself isn't miscounted as stroke boundary). The
/// matte's row/column pixel pitches differ slightly; the result is scaled by
/// their mean since a stroke is locally isotropic.
fn stroke_width_px(
    data: &[u8],
    mw: usize,
    mh: usize,
    sup_top: usize,
    sup_bottom: usize,
    rows_to_px: f32,
    cols_to_px: f32,
) -> f32 {
    // Core threshold relative to this line's own peak alpha, so a faint bold
    // matte keeps its core instead of being thinned by a fixed high cut.
    let peak = (sup_top..=sup_bottom)
        .flat_map(|y| (0..mw).map(move |x| data[y * mw + x]))
        .max()
        .unwrap_or(0);
    let core = ((peak as f32 * STROKE_CORE_FRAC) as u8).max(INK_CUT as u8);
    let ink = |x: usize, y: usize| data[y * mw + x] >= core;
    let mut area: u32 = 0;
    let mut perimeter: u32 = 0;
    for y in sup_top..=sup_bottom {
        for x in 0..mw {
            if !ink(x, y) {
                continue;
            }
            area += 1;
            let up = y > 0 && ink(x, y - 1);
            let down = y + 1 < mh && ink(x, y + 1);
            let left = x > 0 && ink(x - 1, y);
            let right = x + 1 < mw && ink(x + 1, y);
            if !(up && down && left && right) {
                perimeter += 1;
            }
        }
    }
    if area == 0 || perimeter == 0 {
        return 0.0;
    }
    let stroke_matte = 2.0 * area as f32 / perimeter as f32;
    stroke_matte * 0.5 * (rows_to_px + cols_to_px)
}

/// First and last columns (within the support rows) carrying real ink. A column
/// counts as inked when its summed alpha over the band clears a couple of solid
/// texels, so a lone anti-aliased speck doesn't stretch the extent.
fn ink_column_span(data: &[u8], mw: usize, sup_top: usize, sup_bottom: usize) -> (usize, usize) {
    let col_min = 2 * INK_CUT as u32;
    let inked = |x: usize| -> bool {
        (sup_top..=sup_bottom)
            .map(|y| data[y * mw + x] as u32)
            .sum::<u32>()
            >= col_min
    };
    let left = (0..mw).find(|&x| inked(x));
    let right = (0..mw).rev().find(|&x| inked(x));
    match (left, right) {
        (Some(l), Some(r)) => (l, r),
        _ => (0, mw - 1),
    }
}

/// Sub-row position where the cumulative ink mass over `[s, e]` first reaches
/// `frac` of `total`, linearly interpolated within the crossing row. The
/// integral form is what makes the x-height read robust to a single tall row.
fn mass_row(row_sum: &[u32], s: usize, e: usize, total: f64, frac: f32) -> f32 {
    if total <= 0.0 {
        return s as f32;
    }
    let target = total * frac as f64;
    let mut cum = 0.0f64;
    for y in s..=e {
        let r = row_sum[y] as f64;
        if cum + r >= target {
            let f = if r > 0.0 {
                ((target - cum) / r) as f32
            } else {
                0.0
            };
            return y as f32 + f;
        }
        cum += r;
    }
    e as f32
}

/// Pick the contiguous run of above-threshold rows that straddles the strip
/// centre (the box is built around the target line, so its ink sits mid-strip).
/// If no run contains the centre, take the run whose interval is nearest it —
/// that drops off-centre bands from neighbouring-line bleed or stray blobs.
fn central_band(row_sum: &[u32], threshold: f32, mh: usize) -> Option<(usize, usize)> {
    let centre = mh / 2;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for (y, &s) in row_sum.iter().enumerate() {
        let active = s as f32 >= threshold;
        match (active, start) {
            (true, None) => start = Some(y),
            (false, Some(s0)) => {
                runs.push((s0, y - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s0) = start {
        runs.push((s0, mh - 1));
    }
    if runs.is_empty() {
        return None;
    }
    if let Some(&run) = runs.iter().find(|&&(t, b)| centre >= t && centre <= b) {
        return Some(run);
    }
    runs.into_iter().min_by_key(|&(t, b)| {
        let d = if centre < t { t - centre } else { centre - b };
        d
    })
}

/// Fit the baseline slope (matte rows per column) by regressing each column's
/// ink centroid against its column index, with two winsorizing passes so
/// descenders/ascenders that overshoot can't drag the line. Each column's
/// centroid comes from the inked run that overlaps the band centre — not a
/// fixed global window — so a tilted line isn't clipped at the band edges, and
/// a separate run from neighbouring-line bleed is left out. Returns 0 when too
/// few columns carry ink to trust a slope.
fn baseline_slope(data: &[u8], mw: usize, mh: usize, band_top: usize, band_bottom: usize) -> f32 {
    let band_mid = (band_top + band_bottom) / 2;
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    for x in 0..mw {
        if let Some(centroid) = column_band_centroid(data, mw, mh, x, band_mid) {
            xs.push(x as f32);
            ys.push(centroid);
        }
    }
    if xs.len() < MIN_TILT_COLUMNS {
        return 0.0;
    }

    let mut keep = vec![true; xs.len()];
    let mut slope = 0.0;
    for _ in 0..2 {
        let (m, b) = least_squares(&xs, &ys, &keep);
        slope = m;
        let mut resid: Vec<f32> = xs
            .iter()
            .zip(&ys)
            .map(|(&x, &y)| (y - (m * x + b)).abs())
            .collect();
        let mut sorted = resid.clone();
        sorted.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let mad = sorted[sorted.len() / 2].max(1e-3);
        for (k, r) in resid.drain(..).enumerate() {
            keep[k] = r <= 3.0 * mad;
        }
        if keep.iter().filter(|&&k| k).count() < MIN_TILT_COLUMNS {
            break;
        }
    }
    slope
}

/// Alpha-weighted centroid row of the column's inked run that contains (or sits
/// nearest) `band_mid`. Scanning the column's own runs — rather than a fixed row
/// window — means a tilted line's stroke is measured whole at every column, and
/// a disjoint run from a neighbouring line is excluded.
fn column_band_centroid(
    data: &[u8],
    mw: usize,
    mh: usize,
    x: usize,
    band_mid: usize,
) -> Option<f32> {
    let mut best: Option<(usize, f32, f32)> = None; // (dist-to-mid, wsum, ysum)
    let mut run_w: f32 = 0.0;
    let mut run_y: f32 = 0.0;
    let mut run_start: Option<usize> = None;
    let flush =
        |start: usize, end: usize, w: f32, ys: f32, best: &mut Option<(usize, f32, f32)>| {
            let dist = if band_mid < start {
                start - band_mid
            } else if band_mid > end {
                band_mid - end
            } else {
                0
            };
            if best.map_or(true, |(d, _, _)| dist < d) {
                *best = Some((dist, w, ys));
            }
        };
    for y in 0..mh {
        let a = data[y * mw + x] as u16;
        if a >= INK_CUT {
            if run_start.is_none() {
                run_start = Some(y);
                run_w = 0.0;
                run_y = 0.0;
            }
            run_w += a as f32;
            run_y += a as f32 * y as f32;
        } else if let Some(s) = run_start.take() {
            flush(s, y - 1, run_w, run_y, &mut best);
        }
    }
    if let Some(s) = run_start {
        flush(s, mh - 1, run_w, run_y, &mut best);
    }
    best.and_then(|(_, w, ys)| (w > 0.0).then_some(ys / w))
}

/// Ordinary least-squares slope/intercept over the kept points.
fn least_squares(xs: &[f32], ys: &[f32], keep: &[bool]) -> (f32, f32) {
    let n = keep.iter().filter(|&&k| k).count() as f32;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for ((&x, &y), &k) in xs.iter().zip(ys).zip(keep) {
        if !k {
            continue;
        }
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-6 {
        return (0.0, sy / n);
    }
    let m = (n * sxy - sx * sy) / denom;
    let b = (sy - m * sx) / n;
    (m, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Paint a matte where each column's ink spans `[centre(x)-half, centre(x)+half]`
    /// rows at full alpha, with `centre(x) = c0 + slope*x`.
    fn tilted_band(mw: u32, mh: u32, c0: f32, slope: f32, half: f32) -> GrayImage {
        GrayImage::from_fn(mw, mh, |x, y| {
            let c = c0 + slope * x as f32;
            let v = if (y as f32) >= c - half && (y as f32) <= c + half {
                255
            } else {
                0
            };
            image::Luma([v])
        })
    }

    #[test]
    fn flat_band_recovers_height_and_zero_tilt() {
        // Rows 16..=31 inked (16 rows of 48), full width, no tilt.
        let mw = 64;
        let mh = 48;
        let m = tilted_band(mw, mh, 23.5, 0.0, 7.5);
        let r = measure_line(&m, mw as f32, mh as f32).unwrap();
        // 16-row band; the 10–90% mass span is 0.8× of it ≈ 12.8.
        assert!((r.x_height - 12.8).abs() <= 1.0, "x_height {}", r.x_height);
        assert!(
            r.baseline_angle_delta.abs() < 1e-3,
            "tilt {}",
            r.baseline_angle_delta
        );
        // Band centred on the strip ⇒ no centreline offset.
        assert!(
            r.centerline_offset.abs() < 1.0,
            "centerline {}",
            r.centerline_offset
        );
    }

    #[test]
    fn offcentre_band_reports_centerline_offset() {
        // Band rows 30..=39, centred at 34.5 in a 48-row strip (centre 23.5):
        // ~11 rows below centre.
        let mw = 64;
        let mh = 48;
        let m = tilted_band(mw, mh, 34.5, 0.0, 4.5);
        let r = measure_line(&m, mw as f32, mh as f32).unwrap();
        assert!(
            (r.centerline_offset - 11.0).abs() <= 1.5,
            "centerline {}",
            r.centerline_offset
        );
    }

    #[test]
    fn x_height_scales_with_box_height() {
        let mw = 64;
        let mh = 48;
        let m = tilted_band(mw, mh, 23.5, 0.0, 7.5);
        // Box twice as tall as the matte → x-height doubles in image space
        // (12.8 rows → 25.6px).
        let r = measure_line(&m, mw as f32, 2.0 * mh as f32).unwrap();
        assert!((r.x_height - 25.6).abs() <= 2.0, "x_height {}", r.x_height);
    }

    #[test]
    fn recovers_baseline_tilt_sign_and_magnitude() {
        let mw = 64;
        let mh = 48;
        let slope = 0.06; // rows per column
        let m = tilted_band(mw, mh, 20.0, slope, 6.0);
        // Square mapping (box dims == matte dims) ⇒ delta == atan(slope).
        let r = measure_line(&m, mw as f32, mh as f32).unwrap();
        let want = (slope).atan();
        assert!(
            (r.baseline_angle_delta - want).abs() < 0.02,
            "delta {} want {}",
            r.baseline_angle_delta,
            want
        );
    }

    #[test]
    fn central_band_ignores_offcentre_neighbour_bleed() {
        // A faint neighbour band near the top (rows 2..=6) plus the real line
        // band straddling centre (rows 20..=31). The central band must win.
        let mw = 64;
        let mh = 48;
        let m = GrayImage::from_fn(mw, mh, |_x, y| {
            let v = if (20..=31).contains(&y) {
                255
            } else if (2..=6).contains(&y) {
                255
            } else {
                0
            };
            image::Luma([v])
        });
        let r = measure_line(&m, mw as f32, mh as f32).unwrap();
        // 12-row central band; 10–90% mass span ≈ 9.6.
        assert!((r.x_height - 9.6).abs() <= 1.0, "x_height {}", r.x_height);
    }

    #[test]
    fn recovers_horizontal_ink_extent() {
        // Ink only in columns 10..=49 of 64; rows 16..=31. Width is the full
        // column span (40 cols), centred 2 cols right of strip centre.
        let mw = 64u32;
        let mh = 48u32;
        let m = GrayImage::from_fn(mw, mh, |x, y| {
            let on = (10..=49).contains(&x) && (16..=31).contains(&y);
            image::Luma([if on { 255 } else { 0 }])
        });
        let r = measure_line(&m, mw as f32, mh as f32).unwrap();
        assert!((r.width - 40.0).abs() <= 1.0, "width {}", r.width);
        // ink centre column = 30, strip centre = 32 ⇒ ~ -2px offset.
        assert!(
            (r.center_u_offset + 2.0).abs() <= 1.0,
            "center_u {}",
            r.center_u_offset
        );
    }

    #[test]
    fn empty_matte_is_none() {
        let m = GrayImage::from_pixel(32, 48, image::Luma([0]));
        assert!(measure_line(&m, 32.0, 48.0).is_none());
    }
}

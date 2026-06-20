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

/// Matte alpha at or above which a texel counts as ink. The single source of truth for
/// "what is ink"; the erase path's `color_matting::INK_ALPHA_CUT` aliases it.
pub const INK_CUT: u16 = 40;
/// Fraction of the peak row-profile that marks a row as part of the line's
/// vertical *support* (as opposed to inter-line gap or padding). Deliberately
/// low: it only has to separate this line's ink from a neighbouring line that
/// bled into the box's padding (which shows as a *disjoint* run), not to find
/// the x-height edge — that comes from the modal top/bottom-of-ink rows.
const SUPPORT_FRAC: f32 = 0.12;
/// Minimum column span, as a fraction of the support height, for a column to vote
/// in the span-coverage profile. Drops sliver columns — arch middles, round-glyph
/// edge grazes, dots — that span only a fraction of the band; the knee where arch
/// letters recover their true height sits at ~0.15, so this clears it with margin
/// while staying well below the ~0.75 floor of a genuine x-height-only column.
const MIN_SPAN_FRAC: f32 = 0.20;
/// Minimum inked columns to attempt a baseline-tilt fit. Below this the slope
/// is noise; report zero tilt and keep just the x-height.
const MIN_TILT_COLUMNS: usize = 8;
/// Clamp on the recovered tilt. The matte was produced from a strip dewarped
/// with the rough angle, so a real residual is small; a larger fit is a
/// degenerate matte, not a steeper line.
const MAX_TILT_RADIANS: f32 = 0.26; // ~15°
/// The confident stroke *core*: texels at or above this fraction of the line's own peak
/// alpha. Used for both stroke-width geometry and bold pooling (via `stroke_core_cut`) — the
/// feather drops out either way, and gating bold on the core stops anti-aliased edge pixels
/// (low, ambiguous bold values) from dragging the pooled mean down on thick display words.
const STROKE_CORE_FRAC: f32 = 0.6;

/// Per-line core alpha cut from the line's peak matte alpha. One definition of "the stroke
/// core" for stroke-width geometry and bold pooling, floored at `INK_CUT` for a near-empty matte.
pub(crate) fn stroke_core_cut(peak_alpha: u8) -> u8 {
    ((peak_alpha as f32 * STROKE_CORE_FRAC) as u8).max(INK_CUT as u8)
}
/// Need at least this many ink pixels to trust a pooled bold estimate.
pub(crate) const INK_BOLD_MIN_PX: u64 = 30;
/// Mean pooled bold (0..1) at or above which a word/line counts as bold.
pub(crate) const MODEL_BOLD_THRESHOLD: f32 = 0.65;
/// A firing gap wider than this multiple of the line's median character advance starts a
/// new word — the fallback for recognizer models/charsets that under-emit the space class.
/// Above typical kerning and letter-spacing jitter, below a true inter-word gap.
pub(crate) const WORD_GAP_FACTOR: f32 = 1.8;

/// Per-reading-axis-column reduction of an ink strip's bold channel: for each strip column,
/// the matte-gated sum of the bold channel and the count of ink pixels, prefix-summed so any
/// reading-axis fraction window pools in O(1). The strip's height never escapes pooling, so
/// this is all the live acquire stage needs to keep to recover per-word bold at rec time —
/// far smaller than the 2-D strips, and indexed by the same reading-axis fraction the CTC
/// firings use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoldProfile {
    width: u32,
    /// `psum[x]` = Σ bold over ink pixels in columns `[0, x)`; length `width + 1`.
    psum: Vec<u64>,
    /// `pcount[x]` = ink pixel count in columns `[0, x)`; length `width + 1`.
    pcount: Vec<u64>,
}

impl BoldProfile {
    pub fn from_strip(bold: &GrayImage, matte: &GrayImage) -> Option<Self> {
        if bold.dimensions() != matte.dimensions() || matte.width() == 0 {
            return None;
        }
        let (w, h) = matte.dimensions();
        let core = stroke_core_cut(matte.iter().copied().max().unwrap_or(0));
        let mut psum = vec![0u64; w as usize + 1];
        let mut pcount = vec![0u64; w as usize + 1];
        for x in 0..w {
            let (mut s, mut c) = (0u64, 0u64);
            for y in 0..h {
                if matte.get_pixel(x, y)[0] >= core {
                    s += bold.get_pixel(x, y)[0] as u64;
                    c += 1;
                }
            }
            psum[x as usize + 1] = psum[x as usize] + s;
            pcount[x as usize + 1] = pcount[x as usize] + c;
        }
        Some(Self {
            width: w,
            psum,
            pcount,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    /// Mean bold (0..1) over the whole strip's ink, or `None` when there is too little ink to
    /// trust — the whole-line fallback weight, matching [`crate::ppocr::InkStrip::pooled_bold`].
    pub fn whole_pooled_bold(&self) -> Option<f32> {
        let n = self.pcount[self.width as usize];
        (n >= INK_BOLD_MIN_PX).then(|| self.psum[self.width as usize] as f32 / n as f32 / 255.0)
    }

    /// Mean bold probability (0..1) over the ink pixels in the reading-axis fraction window
    /// `[frac_lo, frac_hi)`. 0.0 when the window has no ink.
    fn pool(&self, frac_lo: f32, frac_hi: f32) -> f32 {
        let w = self.width as f32;
        let x0 = (frac_lo * w).floor().clamp(0.0, w) as usize;
        let x1 = (frac_hi * w).ceil().clamp(0.0, w) as usize;
        let n = self.pcount[x1] - self.pcount[x0];
        if n == 0 {
            return 0.0;
        }
        (self.psum[x1] - self.psum[x0]) as f32 / n as f32 / 255.0
    }
}

/// Median reading-axis advance between consecutive firings (robust to the few large
/// inter-word gaps). Returns 1.0 for fewer than two firings, disabling gap splitting.
fn firing_median_advance(firings: &[(char, f32)]) -> f32 {
    if firings.len() < 2 {
        return 1.0;
    }
    let mut adv: Vec<f32> = firings.windows(2).map(|w| w[1].1 - w[0].1).collect();
    adv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    adv[adv.len() / 2]
}

/// Word units (inclusive `(start, end)` firing index ranges over non-whitespace firings),
/// split on the recognizer's space class or a firing gap wider than [`WORD_GAP_FACTOR`]×
/// the median advance.
fn firing_word_units(firings: &[(char, f32)]) -> Vec<(usize, usize)> {
    let gap_thresh = WORD_GAP_FACTOR * firing_median_advance(firings);
    let mut units = Vec::new();
    let mut cur: Option<(usize, usize)> = None;
    for (i, (c, at)) in firings.iter().enumerate() {
        if c.is_whitespace() {
            if let Some((s, l)) = cur.take() {
                units.push((s, l));
            }
            continue;
        }
        cur = match cur {
            None => Some((i, i)),
            Some((s, l)) => {
                if at - firings[l].1 > gap_thresh {
                    units.push((s, l));
                    Some((i, i))
                } else {
                    Some((s, i))
                }
            }
        };
    }
    if let Some((s, l)) = cur {
        units.push((s, l));
    }
    units
}

/// Bold byte ranges within `text`, from CTC firings paired with the ink bold profile. Word
/// units come from the firings (space class / gaps for space scripts, per glyph for CJK);
/// each unit's reading-axis window is pooled over the bold profile and, if it clears
/// `threshold`, the matching text word's byte range is emitted. Units map onto text words
/// positionally; returns empty (caller falls back to a per-line estimate) if the firings
/// are absent or the unit and word counts disagree.
/// Pair CTC firings with `text`'s words. Returns `(firing index ranges, text byte ranges)`,
/// one entry each per word and positionally aligned, or `None` when the counts disagree (RTL
/// or multi-chunk lines whose firings don't line up 1:1 with the text). For CJK each glyph is
/// its own unit; for spaced scripts units come from the recognizer's space class / firing
/// gaps. Shared by [`word_bold_ranges`] and [`firing_word_boxes`].
fn firing_units_and_words(
    text: &str,
    firings: &[(char, f32)],
    is_cjk: bool,
) -> Option<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
    let units: Vec<(usize, usize)> = if is_cjk {
        firings
            .iter()
            .enumerate()
            .filter(|(_, (c, _))| !c.is_whitespace())
            .map(|(i, _)| (i, i))
            .collect()
    } else {
        firing_word_units(firings)
    };
    let base = text.as_ptr() as usize;
    let text_words: Vec<(usize, usize)> = if is_cjk {
        text.char_indices()
            .filter(|(_, c)| !c.is_whitespace())
            .map(|(b, c)| (b, b + c.len_utf8()))
            .collect()
    } else {
        text.split_whitespace()
            .map(|w| {
                let s = w.as_ptr() as usize - base;
                (s, s + w.len())
            })
            .collect()
    };
    (units.len() == text_words.len()).then_some((units, text_words))
}

/// Word spans of `text` as `(char_start, char_end_exclusive, byte_start, byte_end)`: whitespace
/// tokens for spaced scripts, each non-space glyph for CJK. Word segmentation comes from the
/// recognized text itself, so every recognized line yields its words.
fn text_word_spans(text: &str, is_cjk: bool) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    if is_cjk {
        for (ci, (b, c)) in text.char_indices().enumerate() {
            if !c.is_whitespace() {
                out.push((ci, ci + 1, b, b + c.len_utf8()));
            }
        }
        return out;
    }
    let mut word: Option<(usize, usize)> = None; // (char_start, byte_start)
    let mut char_end = 0usize;
    let mut byte_end = 0usize;
    for (ci, (b, c)) in text.char_indices().enumerate() {
        if c.is_whitespace() {
            if let Some((cs, bs)) = word.take() {
                out.push((cs, ci, bs, b));
            }
        } else if word.is_none() {
            word = Some((ci, b));
        }
        char_end = ci + 1;
        byte_end = b + c.len_utf8();
    }
    if let Some((cs, bs)) = word.take() {
        out.push((cs, char_end, bs, byte_end));
    }
    out
}

/// Per-word source boxes in image space. Words are segmented from the recognized `text`, and
/// each word's reading-axis span is read from the CTC `firings`, which are 1:1 with the text by
/// construction (the recognizer's `text` is its decoded chars, and `ppocr` carries the firings
/// through the same trim/normalize). A firing sits near its glyph's *trailing* edge (peaky CTC
/// biases `(t+0.5)/seq_len` ~one stride forward), so glyph `i` spans from the previous firing to
/// its own — `char_edge[i]` is `firings[i-1]` (0 for the first glyph). Returns empty when no
/// aligned firings are available (RTL lines, which carry none). Mapped via [`OrientedRect::subspan`].
pub fn firing_word_boxes(
    text: &str,
    firings: &[(char, f32)],
    is_cjk: bool,
    oriented: &crate::ocr::OrientedRect,
    line_index: u32,
) -> Vec<crate::ocr::PositionedWord> {
    let n = text.chars().count();
    if firings.len() != n || n == 0 {
        return Vec::new();
    }
    let char_edge: Vec<f32> = std::iter::once(0.0)
        .chain(firings.iter().map(|f| f.1.clamp(0.0, 1.0)))
        .collect();
    let w = oriented.width;
    let words = text_word_spans(text, is_cjk);
    let last = words.len().saturating_sub(1);
    words
        .iter()
        .enumerate()
        .map(|(wi, &(cs, ce, bs, be))| {
            let lo = char_edge[cs].min(char_edge[ce]);
            let hi = char_edge[cs].max(char_edge[ce]);
            // A firing isn't guaranteed to land on any particular part of its glyph, so the span
            // under/overshoots the visible word by up to ~a glyph. Pad with a fraction of an
            // average glyph: a quarter on interior sides (so the inter-word space survives —
            // neighbours' pads only meet halfway) and a half on the line's outer ends.
            let glyph = (hi - lo) / (ce - cs).max(1) as f32;
            let left_pad = glyph * if wi == 0 { 0.5 } else { 0.25 };
            let right_pad = glyph * if wi == last { 0.5 } else { 0.25 };
            crate::ocr::PositionedWord {
                text: text[bs..be].to_string(),
                bounds: oriented.subspan(
                    ((lo - left_pad).max(0.0)) * w,
                    ((hi + right_pad).min(1.0)) * w,
                ),
                line_index,
            }
        })
        .collect()
}

pub fn word_bold_ranges(
    text: &str,
    firings: &[(char, f32)],
    is_cjk: bool,
    profile: &BoldProfile,
    threshold: f32,
) -> Vec<(u32, u32)> {
    if firings.is_empty() || profile.width() == 0 {
        return Vec::new();
    }
    let Some((units, text_words)) = firing_units_and_words(text, firings, is_cjk) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (k, &(s, _)) in units.iter().enumerate() {
        let lo = firings[s].1.clamp(0.0, 1.0);
        let hi = if k + 1 < units.len() {
            firings[units[k + 1].0].1.clamp(0.0, 1.0)
        } else {
            1.0
        }
        .max(lo);
        if profile.pool(lo, hi) >= threshold {
            let (bs, be) = text_words[k];
            out.push((bs as u32, be as u32));
        }
    }
    out
}

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
    /// Kept as a `viz_pipeline` diagnostic; bold is now decided by the ink model's
    /// bold channel, not this ratio.
    pub fn weight_ratio(&self) -> f32 {
        self.stroke_width / self.x_height.max(1e-3)
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
    // sits in a disjoint run off-centre).
    let (sup_top, sup_bottom) = central_band(&row_sum, peak as f32 * SUPPORT_FRAC, mh as usize)?;

    // The x-height band is bounded by the half-maximum crossings of the column
    // span-coverage profile. Caps/ascenders pile a shoulder above the x-line and
    // descenders one below the baseline, but each is only a minority of columns so
    // both shoulders sit well under half the peak — the half-max edge therefore
    // tracks the dense x-height band regardless of the word's letter mix, where a
    // column-top mode tips to the caps once they are a plurality (e.g. "MyAppList").
    let span_cov = column_span_coverage(data, mw_us, mh as usize, sup_top, sup_bottom);
    let (xheight_top, baseline_row) = band_edges(&span_cov, sup_top, sup_bottom);

    let rows_to_px = box_height / mh as f32;
    let x_height = (baseline_row - xheight_top).max(0.0) * rows_to_px;
    let band_center = (xheight_top + baseline_row) * 0.5;
    let centerline_offset = (band_center - (mh as f32 - 1.0) * 0.5) * rows_to_px;

    // Horizontal ink extent: the full span of inked columns within the support
    // band — the *full* span, not a percentile, since trimming would re-clip the
    // first/last glyph we're trying to recover.
    let (col_left, col_right) = ink_column_span(data, mw_us, sup_top, sup_bottom);
    let cols_to_px = box_width / mw as f32;
    let width = (col_right - col_left + 1) as f32 * cols_to_px;
    let ink_center_col = (col_left + col_right + 1) as f32 * 0.5;
    let center_u_offset = (ink_center_col - mw as f32 * 0.5) * cols_to_px;

    let slope_rows_per_col = baseline_slope(data, mw_us, sup_top, sup_bottom);
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
    let core = stroke_core_cut(peak);
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

/// Per-row count of columns whose inked span `[top, bottom]` covers the row. By
/// filling each column from its first to its last inked row, interior holes — the
/// bar of an `e`, the bowl of an `o` — are ignored, so a row of aligned horizontal
/// strokes can't spike the profile and collapse the half-max band the way a raw
/// alpha-sum or a per-row inked-column count does. The result is a clean top-hat:
/// nearly every column spans the x-height band, dropping to the ascender/cap
/// fraction above the x-line and the descender fraction below the baseline.
///
/// Columns whose span is under [`MIN_SPAN_FRAC`] of the support height are dropped
/// first: these are slivers — the open middle of an `n`/`m`/`u` arch, the grazing
/// left/right edges of an `o`, a stray dot or hyphen — that read a partial height
/// and would drag the band short. Only full-height columns (stems, letter bodies)
/// vote, so a word of pure arches still recovers its true x-height.
fn column_span_coverage(
    data: &[u8],
    mw: usize,
    mh: usize,
    sup_top: usize,
    sup_bottom: usize,
) -> Vec<u32> {
    let min_span = (MIN_SPAN_FRAC * (sup_bottom - sup_top + 1) as f32).ceil() as usize;
    let mut cov = vec![0u32; mh];
    for x in 0..mw {
        let Some(top) = (sup_top..=sup_bottom).find(|&y| data[y * mw + x] as u16 >= INK_CUT) else {
            continue;
        };
        let bottom = (sup_top..=sup_bottom)
            .rev()
            .find(|&y| data[y * mw + x] as u16 >= INK_CUT)
            .expect("a column with a top has a bottom");
        if bottom - top + 1 < min_span {
            continue;
        }
        for c in cov.iter_mut().take(bottom + 1).skip(top) {
            *c += 1;
        }
    }
    cov
}

/// The x-height line and baseline as the half-maximum crossings of a vertical
/// `profile`, scanning outward from its peak row and linearly interpolating the
/// crossing within the straddling row pair. A density level (not a column count
/// or a cumulative-mass percentile), so the ascender/cap shoulder above the
/// x-line and the descender shoulder below the baseline — each only a fraction of
/// the columns, hence under half the peak — are stepped over rather than included.
fn band_edges(profile: &[u32], sup_top: usize, sup_bottom: usize) -> (f32, f32) {
    let peak_row = (sup_top..=sup_bottom)
        .max_by_key(|&y| profile[y])
        .expect("non-empty support");
    let half = profile[peak_row] as f32 * 0.5;

    let mut y = peak_row;
    while y > sup_top && profile[y] as f32 >= half {
        y -= 1;
    }
    let top = if profile[y] as f32 >= half {
        sup_top as f32
    } else {
        let (a, b) = (profile[y] as f32, profile[y + 1] as f32);
        y as f32 + (half - a) / (b - a)
    };

    let mut y = peak_row;
    while y < sup_bottom && profile[y] as f32 >= half {
        y += 1;
    }
    let bottom = if profile[y] as f32 >= half {
        sup_bottom as f32
    } else {
        let (a, b) = (profile[y] as f32, profile[y - 1] as f32);
        y as f32 - (half - a) / (b - a)
    };

    (top, bottom)
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

/// Fit the baseline slope (matte rows per column) by Theil–Sen regression of each
/// column's bottom-of-ink row against its column index. The bottom edge is the
/// typographic baseline, which caps, ascenders and x-height letters all share, so
/// the fit is immune to the upward reach of tall glyphs that biases a centroid.
/// The few descenders that survive into the support band push their columns low;
/// Theil–Sen's median of pairwise slopes discards them — even clustered at one
/// end of a word, as in "queue" — without picking a side or a trim threshold.
/// Returns 0 when too few columns carry ink to trust a slope.
fn baseline_slope(data: &[u8], mw: usize, support_top: usize, support_bottom: usize) -> f32 {
    // Cap the pairwise work on very wide strips; striding columns barely moves a
    // median-of-slopes fit.
    let stride = (mw / 600).max(1);
    let mut xs: Vec<f32> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    for x in (0..mw).step_by(stride) {
        if let Some(bottom) = column_baseline_row(data, mw, x, support_top, support_bottom) {
            xs.push(x as f32);
            ys.push(bottom);
        }
    }
    if xs.len() < MIN_TILT_COLUMNS {
        return 0.0;
    }
    theil_sen_slope(&xs, &ys)
}

/// Median of all pairwise slopes (Theil–Sen). Robust to ~29% outliers regardless
/// of whether they cluster at one end, so descender columns cannot lean the fit.
fn theil_sen_slope(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len();
    let mut slopes: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = xs[j] - xs[i];
            if dx.abs() < 1e-6 {
                continue;
            }
            slopes.push((ys[j] - ys[i]) / dx);
        }
    }
    if slopes.is_empty() {
        return 0.0;
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = slopes.len() / 2;
    if slopes.len() % 2 == 1 {
        slopes[m]
    } else {
        0.5 * (slopes[m - 1] + slopes[m])
    }
}

/// Lowest inked row in the column, restricted to the line's support band. The
/// support band drops a neighbouring line's bleed (a disjoint run outside the
/// band) and the descenders too sparse to enter the support; taking the *lowest*
/// row — not the run nearest the band centre — reads the true baseline through
/// open/curved glyphs (c, s, e) whose column splits into a top arc and a separate
/// bottom arc, where a nearest-run rule would wrongly latch onto the top arc.
fn column_baseline_row(
    data: &[u8],
    mw: usize,
    x: usize,
    support_top: usize,
    support_bottom: usize,
) -> Option<f32> {
    (support_top..=support_bottom)
        .rev()
        .find(|&y| data[y * mw + x] as u16 >= INK_CUT)
        .map(|y| y as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_bold_ranges_marks_only_the_bold_word() {
        // 100×10 strip, fully inked; the left half is bold (255), the right half is not (0).
        let matte = GrayImage::from_pixel(100, 10, image::Luma([255]));
        let bold = GrayImage::from_fn(100, 10, |x, _| image::Luma([if x < 50 { 255 } else { 0 }]));
        let profile = BoldProfile::from_strip(&bold, &matte).expect("profile");
        // "aa bb": first word fires in the bold half, second word in the plain half.
        let firings = [('a', 0.1), ('a', 0.2), (' ', 0.45), ('b', 0.6), ('b', 0.7)];
        let ranges = word_bold_ranges("aa bb", &firings, false, &profile, MODEL_BOLD_THRESHOLD);
        assert_eq!(ranges, vec![(0, 2)]);
    }

    #[test]
    fn firing_word_boxes_tile_a_horizontal_line() {
        // Axis-aligned line box: 100 wide, centred at x=50, height 10, baseline at y=20.
        let oriented = crate::ocr::OrientedRect {
            cx: 50.0,
            cy: 20.0,
            width: 100.0,
            height: 10.0,
            angle_radians: 0.0,
        };
        // "aa bb": char edges 0,0.1,0.2,0.45,0.6,0.7. Each word pads ½ glyph on its outer end and
        // ¼ glyph on its interior end.
        let firings = [('a', 0.1), ('a', 0.2), (' ', 0.45), ('b', 0.6), ('b', 0.7)];
        let words = firing_word_boxes("aa bb", &firings, false, &oriented, 0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "aa");
        assert_eq!(words[1].text, "bb");
        // "aa" edges 0..0.2, glyph 0.1: left ½ (clamped to 0), right ¼ → 0..0.225 → centre 11.25, width 22.5.
        assert!((words[0].bounds.width - 22.5).abs() < 1e-3);
        assert!((words[0].bounds.cx - 11.25).abs() < 1e-3);
        // "bb" edges 0.45..0.7, glyph 0.125: left ¼, right ½ → 0.41875..0.7625 → centre 59.0625, width 34.375.
        assert!((words[1].bounds.width - 34.375).abs() < 1e-3);
        assert!((words[1].bounds.cx - 59.0625).abs() < 1e-3);
        // Height/baseline are inherited from the line box.
        assert!((words[0].bounds.cy - 20.0).abs() < 1e-3);
        assert!((words[0].bounds.height - 10.0).abs() < 1e-3);
    }

    #[test]
    fn firing_word_boxes_empty_without_aligned_firings() {
        let oriented = crate::ocr::OrientedRect::axis_aligned(crate::ocr::Rect {
            left: 0,
            top: 0,
            right: 100,
            bottom: 10,
        });
        // Firings count != text chars (the RTL case carries none): no per-word positions to
        // invent, so emit nothing rather than guessing.
        assert!(firing_word_boxes("a b c", &[], false, &oriented, 0).is_empty());
        let firings = [('a', 0.1), (' ', 0.4), ('b', 0.7)];
        assert!(firing_word_boxes("a b c", &firings, false, &oriented, 0).is_empty());
    }

    #[test]
    fn bold_profile_pool_matches_a_direct_mean() {
        let matte = GrayImage::from_fn(20, 8, |x, _| {
            image::Luma([if x % 2 == 0 { 255 } else { 0 }])
        });
        let bold = GrayImage::from_pixel(20, 8, image::Luma([200]));
        let profile = BoldProfile::from_strip(&bold, &matte).expect("profile");
        // Only the 10 even (inked) columns contribute; each is bold 200/255.
        assert!((profile.pool(0.0, 1.0) - 200.0 / 255.0).abs() < 1e-6);
        assert_eq!(profile.whole_pooled_bold(), Some(200.0 / 255.0));
    }

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
        // Rows 16..=31 inked ⇒ baseline-to-x-line span of 15 rows.
        assert!((r.x_height - 15.0).abs() <= 1.0, "x_height {}", r.x_height);
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
        // (15 rows → 30px).
        let r = measure_line(&m, mw as f32, 2.0 * mh as f32).unwrap();
        assert!((r.x_height - 30.0).abs() <= 2.0, "x_height {}", r.x_height);
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
        // Central band rows 20..=31 ⇒ baseline-to-x-line span of 11 rows.
        assert!((r.x_height - 11.0).abs() <= 1.0, "x_height {}", r.x_height);
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

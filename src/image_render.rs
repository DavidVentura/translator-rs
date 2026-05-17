//! Image overlay renderer.
//!
//! Consumes a [`PreparedImageOverlay`] (text regions already erased + per-block
//! layout instructions from [`crate::ocr::prepare_overlay_image`]) and renders
//! the translated text back into the raster, doing all the heavy lifting that
//! used to live in callers (`ImagePainting.kt` on Android, `image_ocr.rs` on
//! Linux): script itemization, BiDi resolution, per-run font selection from a
//! [`FontProvider`] chain, OpenType shaping (rustybuzz — Indic conjuncts,
//! Arabic joining, kerning), greedy line-break + fit-to-bounds loop, and
//! glyph rasterization (zeno).
//!
//! The output is fresh RGBA bytes with the translated text drawn over the
//! existing erased background. Foreground colors come from the prepared
//! overlay.

use std::collections::HashMap;
use std::sync::Arc;

use crate::font_provider::{FontHandle, FontProvider, FontRequest};
use crate::ocr::{OverlayLayoutMode, PreparedImageOverlay, PreparedTextBlock};
use crate::script::Script;
use crate::text_runs::{ScriptRun, itemize};

use rustybuzz::ttf_parser;
use rustybuzz::{Direction, Face, UnicodeBuffer};
use unicode_bidi::{BidiInfo, Level};
use zeno::{Command, Format, Mask, PathBuilder};

/// Knobs for [`render_overlay`].
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// BCP-47 language tag of the translated text. Used as a hint when the
    /// provider needs to pick between regional variants of the same script
    /// (e.g. Han: zh-Hans vs ja vs ko).
    pub language: String,
    /// Smallest font size the fit loop is allowed to try, in pixels.
    pub min_font_size_px: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            language: String::new(),
            min_font_size_px: 8.0,
        }
    }
}

#[derive(Debug)]
pub enum RenderError {
    InvalidImage(String),
    NoUsableFont,
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidImage(m) => write!(f, "invalid image: {m}"),
            Self::NoUsableFont => write!(f, "no usable font from provider"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Rasterize translated text onto `prepared.rgba_bytes` and return the new
/// buffer. The input buffer is treated as 4-byte little-endian ARGB pixels
/// (i.e. the same layout `crate::ocr` produces).
pub fn render_overlay(
    prepared: &PreparedImageOverlay,
    fonts: &dyn FontProvider,
    opts: &RenderOptions,
) -> Result<Vec<u8>, RenderError> {
    let expected = prepared.width as usize * prepared.height as usize * 4;
    if prepared.rgba_bytes.len() != expected {
        return Err(RenderError::InvalidImage(format!(
            "expected {expected} bytes, got {}",
            prepared.rgba_bytes.len()
        )));
    }

    let mut canvas = prepared.rgba_bytes.clone();
    let mut cache = FontCache::default();

    for block in &prepared.blocks {
        if block.translated_text.trim().is_empty() {
            continue;
        }
        match block.layout_hints.layout_mode {
            OverlayLayoutMode::PerLine => {
                render_per_line(&mut canvas, prepared, block, &mut cache, fonts, opts)
            }
            OverlayLayoutMode::BlockRect => {
                render_block_rect(&mut canvas, prepared, block, &mut cache, fonts, opts)
            }
        }
    }

    Ok(canvas)
}

// ---------------------------------------------------------------------------
// Font cache

#[derive(Default)]
struct FontCache {
    fonts: HashMap<FontHandle, Option<Arc<Vec<u8>>>>,
    chains: HashMap<(Script, bool, bool, bool), Vec<FontHandle>>,
}

impl FontCache {
    fn chain_for(
        &mut self,
        script: Script,
        bold: bool,
        italic: bool,
        monospace: bool,
        language: &str,
        fonts: &dyn FontProvider,
    ) -> &[FontHandle] {
        let key = (script, bold, italic, monospace);
        self.chains.entry(key).or_insert_with(|| {
            fonts.locate(&FontRequest {
                script,
                language: language.to_string(),
                bold,
                italic,
                monospace,
            })
        })
    }

    fn bytes(&mut self, handle: &FontHandle) -> Option<Arc<Vec<u8>>> {
        if let Some(slot) = self.fonts.get(handle) {
            return slot.clone();
        }
        let loaded = std::fs::read(&handle.path).ok().map(Arc::new);
        self.fonts.insert(handle.clone(), loaded.clone());
        loaded
    }
}

// ---------------------------------------------------------------------------
// Script-run + run-direction segmentation

#[derive(Debug, Clone)]
struct DirRun {
    /// Byte offsets into the source string.
    start: usize,
    end: usize,
    script: Script,
    rtl: bool,
    /// Position in the laid-out (visual) sequence. For LTR-only text this is
    /// the same as logical order.
    visual_index: usize,
}

/// Itemize `text` into runs that share both script and BiDi direction. The
/// returned vec is ordered logically; `visual_index` indicates the visual
/// order if a BiDi shuffle is needed.
fn segment_runs(text: &str) -> Vec<DirRun> {
    let bidi = BidiInfo::new(text, None);
    let script_runs = itemize(text);

    if bidi.paragraphs.is_empty() {
        return script_runs
            .into_iter()
            .enumerate()
            .map(|(i, r)| DirRun {
                start: r.start,
                end: r.end,
                script: r.script,
                rtl: r.script.is_rtl(),
                visual_index: i,
            })
            .collect();
    }

    let mut out: Vec<DirRun> = Vec::new();
    for para in &bidi.paragraphs {
        let para_range = para.range.clone();
        let para_runs: Vec<&ScriptRun> = script_runs
            .iter()
            .filter(|r| r.start < para_range.end && r.end > para_range.start)
            .collect();

        let mut split: Vec<DirRun> = Vec::new();
        for r in para_runs {
            let from = r.start.max(para_range.start);
            let to = r.end.min(para_range.end);
            split.extend(split_by_bidi_level(text, &bidi.levels, from, to, r.script));
        }

        // Reorder by visual position — unicode_bidi gives us the visual order
        // of the levels per paragraph.
        let (levels, level_runs) = bidi.visual_runs(para, para_range.clone());
        let mut visual_order: Vec<usize> = (0..split.len()).collect();
        visual_order.sort_by_key(|&i| {
            // find the visual index of this run's start byte.
            level_runs
                .iter()
                .position(|lr| lr.start <= split[i].start && split[i].start < lr.end)
                .unwrap_or(0)
        });
        let _ = levels;
        for (visual_index, logical_index) in visual_order.iter().enumerate() {
            let mut run = split[*logical_index].clone();
            run.visual_index = out.len() + visual_index;
            out.push(run);
        }
        // out's logical order is currently what we just appended; sort visual
        // ordering preserved via visual_index.
        // (We intentionally append in logical order for downstream cluster
        // stability; visual_index is what the renderer uses to lay them out.)
    }
    // The previous block rebuilt entries; ensure logical order is by `start`.
    out.sort_by_key(|r| r.start);
    out
}

fn split_by_bidi_level(
    _text: &str,
    levels: &[Level],
    from: usize,
    to: usize,
    script: Script,
) -> Vec<DirRun> {
    if from >= to {
        return Vec::new();
    }
    let mut runs: Vec<DirRun> = Vec::new();
    let mut cursor = from;
    let mut current_level = levels.get(from).copied().unwrap_or_else(Level::ltr);
    for i in (from + 1)..to {
        let lvl = levels.get(i).copied().unwrap_or(current_level);
        if lvl != current_level {
            runs.push(DirRun {
                start: cursor,
                end: i,
                script,
                rtl: current_level.is_rtl(),
                visual_index: 0,
            });
            cursor = i;
            current_level = lvl;
        }
    }
    runs.push(DirRun {
        start: cursor,
        end: to,
        script,
        rtl: current_level.is_rtl(),
        visual_index: 0,
    });
    runs
}

// ---------------------------------------------------------------------------
// Shaping

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ShapedGlyph {
    /// Glyph ID in the font.
    gid: u16,
    /// Horizontal advance in font units.
    advance_x: i32,
    /// X offset from cursor in font units.
    offset_x: i32,
    /// Y offset from baseline in font units.
    offset_y: i32,
    /// Original cluster (byte offset in the run text) — kept for line breaking.
    cluster: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ShapedRun {
    glyphs: Vec<ShapedGlyph>,
    units_per_em: i32,
    ascent: i32,
    descent: i32,
    /// Font bytes shared via Arc so multiple shapes can reuse it.
    font_bytes: Arc<Vec<u8>>,
    ttc_index: u32,
    rtl: bool,
}

fn shape_run(
    text: &str,
    run: &DirRun,
    handle: &FontHandle,
    cache: &mut FontCache,
) -> Option<ShapedRun> {
    let bytes = cache.bytes(handle)?;
    let face = Face::from_slice(bytes.as_slice(), handle.ttc_index)?;
    let units_per_em = face.units_per_em();
    let ascent = face.ascender() as i32;
    let descent = face.descender() as i32;

    let mut buf = UnicodeBuffer::new();
    buf.push_str(&text[run.start..run.end]);
    buf.set_direction(if run.rtl {
        Direction::RightToLeft
    } else {
        Direction::LeftToRight
    });
    buf.set_script(map_script(run.script));
    let glyph_buffer = rustybuzz::shape(&face, &[], buf);
    let infos = glyph_buffer.glyph_infos();
    let positions = glyph_buffer.glyph_positions();

    let mut glyphs = Vec::with_capacity(infos.len());
    for (info, pos) in infos.iter().zip(positions.iter()) {
        glyphs.push(ShapedGlyph {
            gid: info.glyph_id as u16,
            advance_x: pos.x_advance,
            offset_x: pos.x_offset,
            offset_y: pos.y_offset,
            cluster: info.cluster,
        });
    }

    Some(ShapedRun {
        glyphs,
        units_per_em,
        ascent,
        descent,
        font_bytes: bytes,
        ttc_index: handle.ttc_index,
        rtl: run.rtl,
    })
}

fn map_script(s: Script) -> rustybuzz::Script {
    use rustybuzz::script;
    match s {
        Script::Latin => script::LATIN,
        Script::Cyrillic => script::CYRILLIC,
        Script::Greek => script::GREEK,
        Script::Armenian => script::ARMENIAN,
        Script::Hebrew => script::HEBREW,
        Script::Arabic => script::ARABIC,
        Script::Devanagari => script::DEVANAGARI,
        Script::Bengali => script::BENGALI,
        Script::Gurmukhi => script::GURMUKHI,
        Script::Gujarati => script::GUJARATI,
        Script::Oriya => script::ORIYA,
        Script::Tamil => script::TAMIL,
        Script::Telugu => script::TELUGU,
        Script::Kannada => script::KANNADA,
        Script::Malayalam => script::MALAYALAM,
        Script::Sinhala => script::SINHALA,
        Script::Thai => script::THAI,
        Script::Lao => script::LAO,
        Script::Tibetan => script::TIBETAN,
        Script::Myanmar => script::MYANMAR,
        Script::Georgian => script::GEORGIAN,
        Script::Ethiopic => script::ETHIOPIC,
        Script::Khmer => script::KHMER,
        Script::Han => script::HAN,
        Script::Hiragana => script::HIRAGANA,
        Script::Katakana => script::KATAKANA,
        Script::Hangul => script::HANGUL,
        Script::Common | Script::Inherited | Script::Other => script::COMMON,
    }
}

// ---------------------------------------------------------------------------
// Per-run pick from the FontProvider chain

/// A contiguous byte range within a run's source text, marked as either
/// "primary font covered it" or ".notdef — needs the next font in the chain".
#[derive(Debug)]
struct FallbackSegment {
    /// Absolute byte offset in the full `text`.
    byte_start: usize,
    /// Absolute byte offset in the full `text` (exclusive).
    byte_end: usize,
    has_real_glyph: bool,
}

/// Group `glyphs` by cluster, mark each cluster as covered or .notdef, and
/// emit byte segments in source order. Adjacent clusters with the same state
/// are merged. Cluster fields are offsets relative to the rustybuzz input
/// (i.e. relative to `run_start`); we rebase to absolute offsets in `text`.
fn compute_fallback_segments(
    glyphs: &[ShapedGlyph],
    run_start: usize,
    run_end: usize,
) -> Vec<FallbackSegment> {
    use std::collections::BTreeMap;
    if glyphs.is_empty() {
        return Vec::new();
    }

    let mut cluster_all_notdef: BTreeMap<u32, bool> = BTreeMap::new();
    for g in glyphs {
        let entry = cluster_all_notdef.entry(g.cluster).or_insert(true);
        if g.gid != 0 {
            *entry = false;
        }
    }

    let clusters: Vec<(u32, bool)> = cluster_all_notdef.into_iter().collect();
    let run_len = run_end - run_start;

    let mut out: Vec<FallbackSegment> = Vec::with_capacity(clusters.len());
    for (i, (cluster_local_start, all_notdef)) in clusters.iter().enumerate() {
        let cluster_local_end = clusters
            .get(i + 1)
            .map(|(c, _)| *c as usize)
            .unwrap_or(run_len);
        let byte_start = run_start + *cluster_local_start as usize;
        let byte_end = run_start + cluster_local_end;
        let has_real_glyph = !*all_notdef;

        if let Some(last) = out.last_mut()
            && last.has_real_glyph == has_real_glyph
            && last.byte_end == byte_start
        {
            last.byte_end = byte_end;
            continue;
        }
        out.push(FallbackSegment {
            byte_start,
            byte_end,
            has_real_glyph,
        });
    }

    out
}

/// Shape `run` against `chain` with per-cluster fallback: if the primary
/// font produces .notdef glyphs for some clusters, those clusters' source
/// bytes are re-shaped against `chain[1..]` and stitched into the output.
///
/// Returns shaped pieces in **visual order** (LTR = source order, RTL = source
/// order reversed). Multiple pieces may share fonts; the caller treats them
/// as a flat sequence to draw with the cursor advancing.
fn pick_handle_and_shape(
    text: &str,
    run: &DirRun,
    chain: &[FontHandle],
    cache: &mut FontCache,
) -> Vec<ShapedRun> {
    if chain.is_empty() {
        return Vec::new();
    }

    let primary = match shape_run(text, run, &chain[0], cache) {
        Some(s) => s,
        None => return pick_handle_and_shape(text, run, &chain[1..], cache),
    };

    let has_notdef = primary.glyphs.iter().any(|g| g.gid == 0);
    if !has_notdef || chain.len() == 1 {
        return vec![primary];
    }

    let segments = compute_fallback_segments(&primary.glyphs, run.start, run.end);
    if segments.is_empty() {
        return vec![primary];
    }

    let mut out: Vec<ShapedRun> = Vec::with_capacity(segments.len());
    for seg in &segments {
        let sub_run = DirRun {
            start: seg.byte_start,
            end: seg.byte_end,
            script: run.script,
            rtl: run.rtl,
            visual_index: run.visual_index,
        };
        if seg.has_real_glyph {
            // Primary font handles this segment; re-shape it in isolation.
            // Cross-segment contextual shaping isn't lost because the
            // segment boundary lies at clusters the primary font already
            // failed on, which by definition broke any shaping context.
            if let Some(s) = shape_run(text, &sub_run, &chain[0], cache) {
                out.push(s);
            }
        } else {
            let fallback = pick_handle_and_shape(text, &sub_run, &chain[1..], cache);
            if !fallback.is_empty() {
                out.extend(fallback);
            } else if let Some(s) = shape_run(text, &sub_run, &chain[0], cache) {
                // No fallback usable — keep tofu from the primary font.
                out.push(s);
            }
        }
    }

    if out.is_empty() {
        return vec![primary];
    }

    if run.rtl {
        out.reverse();
    }

    out
}

// ---------------------------------------------------------------------------
// Layout — PerLine

struct LineShape {
    /// Shaped runs in visual order, each annotated with its width-at-1.0-fontsize.
    runs: Vec<ShapedRun>,
    /// Sum of glyph advances across runs, in font units of each run's font.
    /// Width at a given font size in pixels = sum(advance_units / units_per_em * font_size).
    /// We carry per-run total advances and per-run units_per_em to compute it.
    total_widths: Vec<f32>,
    /// Maximum (ascent / units_per_em) across runs.
    max_ascent_em: f32,
    /// Maximum (-descent / units_per_em) across runs.
    max_descent_em: f32,
}

fn shape_line(
    text: &str,
    chain_lookup: &mut dyn FnMut(Script, &mut FontCache) -> Vec<FontHandle>,
    cache: &mut FontCache,
) -> LineShape {
    let runs = segment_runs(text);
    let mut shaped: Vec<(usize, ShapedRun)> = Vec::new();
    for run in &runs {
        let chain = chain_lookup(run.script, cache);
        for piece in pick_handle_and_shape(text, run, &chain, cache) {
            shaped.push((run.visual_index, piece));
        }
    }
    shaped.sort_by_key(|(vi, _)| *vi);

    let mut total_widths = Vec::with_capacity(shaped.len());
    let mut max_ascent_em: f32 = 0.0;
    let mut max_descent_em: f32 = 0.0;
    let mut runs_only = Vec::with_capacity(shaped.len());
    for (_, run) in shaped {
        let upem = run.units_per_em as f32;
        let total_units: i64 = run.glyphs.iter().map(|g| g.advance_x as i64).sum();
        total_widths.push(total_units as f32 / upem);
        max_ascent_em = max_ascent_em.max(run.ascent as f32 / upem);
        max_descent_em = max_descent_em.max(-(run.descent as f32) / upem);
        runs_only.push(run);
    }

    LineShape {
        runs: runs_only,
        total_widths,
        max_ascent_em,
        max_descent_em,
    }
}

impl LineShape {
    fn width_px(&self, font_size: f32) -> f32 {
        self.total_widths.iter().sum::<f32>() * font_size
    }
    fn line_height_px(&self, font_size: f32) -> f32 {
        (self.max_ascent_em + self.max_descent_em) * font_size
    }
    fn ascent_px(&self, font_size: f32) -> f32 {
        self.max_ascent_em * font_size
    }
}

fn render_per_line(
    canvas: &mut [u8],
    prepared: &PreparedImageOverlay,
    block: &PreparedTextBlock,
    cache: &mut FontCache,
    fonts: &dyn FontProvider,
    opts: &RenderOptions,
) {
    // The block's lines are pre-broken by the OCR side. Treat the block's
    // translated text as a single string that we re-flow across that many
    // line slots, fitting widths.
    let translated = block.translated_text.trim();
    if translated.is_empty() {
        return;
    }

    let language = opts.language.clone();
    let mut size = block
        .layout_hints
        .suggested_font_size_px
        .max(opts.min_font_size_px);

    // Helper to shape the whole translated string for a given font size.
    // Shaping is size-independent (we shape once at upem and scale), so we
    // can reuse this across the size-shrink loop.
    let shaped_full = {
        let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
            c.chain_for(script, false, false, false, &language, fonts)
                .to_vec()
        };
        let mut chain_fn = chain_fn;
        shape_line(translated, &mut chain_fn, cache)
    };

    if shaped_full.runs.is_empty() {
        return;
    }

    // Greedy break: try to assign words from `translated` to each block
    // line such that each line's shaped width fits its target box. If any
    // line overflows, shrink size by 1 and retry. We use the oriented_box's width
    // (reading-direction extent) rather than the AABB width — for tilted text the
    // AABB is wider than the actual line and would let us pack more glyphs than fit.
    let target_widths: Vec<f32> = block.lines.iter().map(|l| l.oriented_box.width).collect();

    let lines_text: Option<Vec<String>> = loop {
        match break_into_lines_by_words(translated, &shaped_full, size, &target_widths) {
            Some(v) => break Some(v),
            None if size > opts.min_font_size_px => {
                size -= 1.0;
                continue;
            }
            None => break None,
        }
    };

    let Some(lines_text) = lines_text else {
        return;
    };

    for (line_text, prepared_line) in lines_text.iter().zip(block.lines.iter()) {
        if line_text.trim().is_empty() {
            continue;
        }
        let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
            c.chain_for(script, false, false, false, &language, fonts)
                .to_vec()
        };
        let mut chain_fn = chain_fn;
        let line_shape = shape_line(line_text, &mut chain_fn, cache);
        if line_shape.runs.is_empty() {
            continue;
        }
        // Origin in image space at line-local cursor=0 along the baseline. In line-local
        // coords this point is at u=-width/2 (left edge); the v coord is chosen so the
        // glyph mass (ascent + descent) is centered on the rect's centre. For a rect
        // whose height matches the font size this collapses to the previous
        // "baseline at rect.top + ascent_px" placement (since
        // (ascent - descent) / 2 == -half_h + ascent when half_h == (ascent+descent)/2),
        // so the PDF erase-replace path is unchanged. For oversized rects (the live
        // overlay path inflates `oriented.height` to leave halo room) the glyph is
        // centred instead of top-aligned.
        let oriented = prepared_line.oriented_box;
        let cos = oriented.angle_radians.cos();
        let sin = oriented.angle_radians.sin();
        let half_w = oriented.width * 0.5;
        let ascent_px = line_shape.ascent_px(size);
        let descent_px = (line_shape.line_height_px(size) - ascent_px).max(0.0);
        let v_from_center = (ascent_px - descent_px) * 0.5;
        // perp_down direction (line-local +v) in image space is (-sin, cos).
        let origin_x = oriented.cx - half_w * cos + v_from_center * (-sin);
        let origin_y = oriented.cy - half_w * sin + v_from_center * cos;
        draw_shaped_line(
            canvas,
            prepared.width,
            prepared.height,
            &line_shape,
            origin_x,
            origin_y,
            cos,
            sin,
            size,
            prepared_line.foreground_argb,
        );
    }
}

/// Split `text` into per-line slices that fit within `target_widths` at the
/// chosen `font_size`. Returns `None` if the text doesn't fit even greedily
/// at the given size.
fn break_into_lines_by_words(
    text: &str,
    shaped: &LineShape,
    font_size: f32,
    target_widths: &[f32],
) -> Option<Vec<String>> {
    if target_widths.is_empty() {
        return None;
    }
    if text.chars().all(|c| !c.is_whitespace()) {
        // No spaces — single-line languages or one big run. Fits only if
        // the full shaped width fits the first line.
        let w = shaped.width_px(font_size);
        if w <= target_widths[0] {
            let mut out: Vec<String> = vec![text.to_string()];
            for _ in 1..target_widths.len() {
                out.push(String::new());
            }
            return Some(out);
        }
        return None;
    }

    // Per-character pixel-width lookup using the shaped result. Fall back
    // proportionally if shape didn't cover (shouldn't happen for our text).
    let upem_widths: f32 = shaped.total_widths.iter().sum();
    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len() as f32;
    let avg_char_em = if total_chars > 0.0 {
        upem_widths / total_chars
    } else {
        0.0
    };
    let measure = |slice: &str| -> f32 {
        // Crude but consistent with shaped totals: count chars * average em
        // width * font_size. Good enough for whitespace-driven greedy break;
        // exact widths come back when we re-shape the assigned slice for
        // drawing.
        slice.chars().count() as f32 * avg_char_em * font_size
    };

    let mut out: Vec<String> = Vec::with_capacity(target_widths.len());
    let mut cursor = 0;
    let bytes_text = text;
    for (idx, &target_w) in target_widths.iter().enumerate() {
        let is_last = idx + 1 == target_widths.len();
        if cursor >= bytes_text.len() {
            out.push(String::new());
            continue;
        }
        let remainder = &bytes_text[cursor..];
        if measure(remainder) <= target_w {
            out.push(remainder.to_string());
            cursor = bytes_text.len();
            continue;
        }
        if is_last {
            return None;
        }

        // Walk word boundaries (spaces) to find the largest prefix that fits.
        let mut last_good_end: Option<usize> = None;
        let mut byte_idx = 0;
        for (off, ch) in remainder.char_indices() {
            if ch == ' ' {
                let candidate_end = off; // exclusive
                let candidate = &remainder[..candidate_end];
                if measure(candidate) <= target_w {
                    last_good_end = Some(candidate_end);
                } else {
                    break;
                }
            }
            byte_idx = off + ch.len_utf8();
        }
        let _ = byte_idx;

        let Some(end) = last_good_end else {
            return None;
        };
        let assigned = &remainder[..end];
        out.push(assigned.to_string());
        // Skip the space we broke at.
        cursor += end;
        while cursor < bytes_text.len() && bytes_text.as_bytes()[cursor] == b' ' {
            cursor += 1;
        }
    }

    if cursor < bytes_text.len() {
        return None;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Layout — BlockRect

fn render_block_rect(
    canvas: &mut [u8],
    prepared: &PreparedImageOverlay,
    block: &PreparedTextBlock,
    cache: &mut FontCache,
    fonts: &dyn FontProvider,
    opts: &RenderOptions,
) {
    let translated = block.translated_text.trim();
    if translated.is_empty() {
        return;
    }
    let language = opts.language.clone();
    let mut size = block
        .layout_hints
        .suggested_font_size_px
        .max(opts.min_font_size_px);

    let bw = block
        .bounding_box
        .right
        .saturating_sub(block.bounding_box.left) as f32;
    let bh = block
        .bounding_box
        .bottom
        .saturating_sub(block.bounding_box.top) as f32;
    if bw <= 0.0 || bh <= 0.0 {
        return;
    }

    let lines = loop {
        let candidate = wrap_into_block(translated, bw, size);
        let line_h = estimate_line_height(translated, size, &language, cache, fonts);
        if line_h <= 0.0 {
            return;
        }
        let max_lines = (bh / line_h).floor() as usize;
        if candidate.len() <= max_lines.max(1) && all_lines_fit(&candidate, bw, size) {
            break candidate;
        }
        if size <= opts.min_font_size_px {
            return;
        }
        size -= 1.0;
    };

    let line_h = estimate_line_height(translated, size, &language, cache, fonts);
    let mut baseline_y = block.bounding_box.top as f32 + line_h * 0.8;
    for line_text in lines {
        if line_text.trim().is_empty() {
            baseline_y += line_h;
            continue;
        }
        let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
            c.chain_for(script, false, false, false, &language, fonts)
                .to_vec()
        };
        let mut chain_fn = chain_fn;
        let line_shape = shape_line(&line_text, &mut chain_fn, cache);
        if line_shape.runs.is_empty() {
            baseline_y += line_h;
            continue;
        }
        // Block-rect layout (CJK vertical / multi-line block) never rotates — block boxes are
        // axis-aligned. Pass identity rotation (cos=1, sin=0).
        draw_shaped_line(
            canvas,
            prepared.width,
            prepared.height,
            &line_shape,
            block.bounding_box.left as f32,
            baseline_y,
            1.0,
            0.0,
            size,
            block.foreground_argb,
        );
        baseline_y += line_h;
    }
}

fn estimate_line_height(
    text: &str,
    size: f32,
    language: &str,
    cache: &mut FontCache,
    fonts: &dyn FontProvider,
) -> f32 {
    let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
        c.chain_for(script, false, false, false, language, fonts)
            .to_vec()
    };
    let mut chain_fn = chain_fn;
    let probe = shape_line(text, &mut chain_fn, cache);
    probe.line_height_px(size).max(size * 1.2)
}

fn all_lines_fit(lines: &[String], width: f32, font_size: f32) -> bool {
    // Re-measure with whitespace-driven greedy width. Conservative — we use
    // average width per char via the global string for parity with
    // break_into_lines_by_words.
    let avg_char_em = 0.5; // generic Latin-ish fallback if we lose context.
    lines
        .iter()
        .all(|l| (l.chars().count() as f32) * avg_char_em * font_size <= width)
}

fn wrap_into_block(text: &str, width: f32, font_size: f32) -> Vec<String> {
    // Whitespace-greedy break against an avg-char-em estimate. Same caveat
    // as PerLine: exact widths are realized when we re-shape each line for
    // drawing.
    let avg_char_em = 0.5;
    let max_chars = ((width / (font_size * avg_char_em)).floor() as usize).max(1);
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;
    for word in text.split_whitespace() {
        let wlen = word.chars().count();
        let with_space = if current.is_empty() { wlen } else { wlen + 1 };
        if current_chars + with_space <= max_chars || current.is_empty() {
            if !current.is_empty() {
                current.push(' ');
                current_chars += 1;
            }
            current.push_str(word);
            current_chars += wlen;
        } else {
            out.push(std::mem::take(&mut current));
            current.push_str(word);
            current_chars = wlen;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

// ---------------------------------------------------------------------------
// Draw a shaped line into the BGRA canvas.

fn draw_shaped_line(
    canvas: &mut [u8],
    width: u32,
    height: u32,
    line: &LineShape,
    origin_x: f32,
    origin_y: f32,
    cos_angle: f32,
    sin_angle: f32,
    font_size: f32,
    fg_argb: u32,
) {
    // cursor_x advances in line-local space (along the reading direction). The image-space
    // pen position for each glyph is obtained by rotating the local pen offset by the line's
    // angle and adding the origin.
    let mut cursor_x = 0.0f32;
    for run in &line.runs {
        let scale = font_size / run.units_per_em as f32;
        let face = match Face::from_slice(run.font_bytes.as_slice(), run.ttc_index) {
            Some(f) => f,
            None => continue,
        };
        for glyph in &run.glyphs {
            // Pen position for this glyph in line-local coords (font y points up; flip).
            let pen_local_x = cursor_x + glyph.offset_x as f32 * scale;
            let pen_local_y = -(glyph.offset_y as f32) * scale;
            let glyph_x = origin_x + pen_local_x * cos_angle - pen_local_y * sin_angle;
            let glyph_y = origin_y + pen_local_x * sin_angle + pen_local_y * cos_angle;

            let mut commands: Vec<Command> = Vec::new();
            let mut sink = OutlineSink {
                builder: &mut commands,
                origin_x: glyph_x,
                origin_y: glyph_y,
                scale,
                cos_angle,
                sin_angle,
            };
            if face
                .outline_glyph(ttf_parser::GlyphId(glyph.gid), &mut sink)
                .is_some()
            {
                let (mask, placement) = Mask::new(commands.as_slice())
                    .format(Format::Alpha)
                    .render();
                blit_mask(
                    canvas,
                    width,
                    height,
                    &mask,
                    placement.left,
                    placement.top,
                    placement.width,
                    placement.height,
                    fg_argb,
                );
            }

            cursor_x += glyph.advance_x as f32 * scale;
        }
    }
}

struct OutlineSink<'a> {
    builder: &'a mut Vec<Command>,
    origin_x: f32,
    origin_y: f32,
    scale: f32,
    /// Rotation of the line's reading direction from the image's +x axis, encoded as
    /// (cos θ, sin θ). For horizontal text these are (1, 0) and `px()` collapses to the
    /// original "translate + y-flip" transform.
    cos_angle: f32,
    sin_angle: f32,
}

impl OutlineSink<'_> {
    fn px(&self, x: f32, y: f32) -> (f32, f32) {
        // Scale glyph contour from font units into pixels; flip y because font coords have y
        // pointing up but image coords have y pointing down.
        let lx = x * self.scale;
        let ly = -y * self.scale;
        // Rotate the local offset into image space by the line's angle, then translate to the
        // glyph's pen origin.
        (
            self.origin_x + lx * self.cos_angle - ly * self.sin_angle,
            self.origin_y + lx * self.sin_angle + ly * self.cos_angle,
        )
    }
}

impl ttf_parser::OutlineBuilder for OutlineSink<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let (px, py) = self.px(x, y);
        self.builder.move_to([px, py]);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let (px, py) = self.px(x, y);
        self.builder.line_to([px, py]);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (cx, cy) = self.px(x1, y1);
        let (px, py) = self.px(x, y);
        self.builder.quad_to([cx, cy], [px, py]);
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (c1x, c1y) = self.px(x1, y1);
        let (c2x, c2y) = self.px(x2, y2);
        let (px, py) = self.px(x, y);
        self.builder.curve_to([c1x, c1y], [c2x, c2y], [px, py]);
    }
    fn close(&mut self) {
        self.builder.close();
    }
}

fn blit_mask(
    canvas: &mut [u8],
    width: u32,
    height: u32,
    mask: &[u8],
    placement_left: i32,
    placement_top: i32,
    mask_w: u32,
    mask_h: u32,
    fg_argb: u32,
) {
    if mask_w == 0 || mask_h == 0 {
        return;
    }
    let fg_bytes = fg_argb.to_ne_bytes();
    for my in 0..mask_h {
        let py = placement_top + my as i32;
        if py < 0 || py >= height as i32 {
            continue;
        }
        for mx in 0..mask_w {
            let px = placement_left + mx as i32;
            if px < 0 || px >= width as i32 {
                continue;
            }
            let m_idx = (my * mask_w + mx) as usize;
            let cov = mask[m_idx];
            if cov == 0 {
                continue;
            }
            let a = cov as f32 / 255.0;
            let inv = 1.0 - a;
            let buf_idx = ((py as u32 * width + px as u32) * 4) as usize;
            for c in 0..3 {
                let blended = fg_bytes[c] as f32 * a + canvas[buf_idx + c] as f32 * inv;
                canvas[buf_idx + c] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

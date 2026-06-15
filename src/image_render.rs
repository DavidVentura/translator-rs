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
use crate::text_shape::{self, DirRun, ShapedGlyph, segment_runs};

use rustybuzz::Face;
use rustybuzz::ttf_parser;
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
        let mut sink = GlyphSink::Canvas {
            canvas: &mut canvas,
            width: prepared.width,
            height: prepared.height,
        };
        match block.layout_hints.layout_mode {
            OverlayLayoutMode::PerLine => {
                render_per_line(&mut sink, block, &mut cache, fonts, opts)
            }
            OverlayLayoutMode::VerticalBlockRect => {
                render_vertical_block_rect(&mut sink, block, &mut cache, fonts, opts)
            }
        }
    }

    Ok(canvas)
}

/// `(font_id, glyph_id, size_px)` — the dense key for the CPU glyph atlas and the
/// GPU atlas. All three dimensions must match for a mask to be reused.
pub type GlyphKey = (u32, u16, u32);

/// One glyph draw call for the GPU atlas path. `pen_x`/`pen_y` are the pen
/// position in canvas-texel coords (pre-rounding, subpixel). `cos`/`sin` are the
/// line's reading-direction angle; for axis-aligned text these are 1.0/0.0. The
/// GPU quad is placed at `pen + rotate(left, top)` and rotated by (cos, sin), so
/// the same upright atlas entry serves both horizontal and tilted text.
#[derive(Clone)]
pub struct GlyphInstanceData {
    pub key: GlyphKey,
    pub pen_x: f32,
    pub pen_y: f32,
    pub cos: f32,
    pub sin: f32,
    pub color: [u8; 4],
}

/// Upright coverage mask for one glyph, copied out of the [`FontCache`] atlas for
/// upload to the GPU atlas. Stored alongside its pen-relative placement so the GPU
/// quad can position it without re-reading the CPU mask.
#[derive(Clone)]
pub struct GlyphMaskData {
    pub key: GlyphKey,
    pub cov: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub left: i32,
    pub top: i32,
}

/// Accumulates glyph instances + unique mask data for GPU atlas upload. Produced
/// by [`collect_overlay_glyphs`] and consumed by the GL compositor.
#[derive(Default)]
pub struct GlyphCollector {
    pub instances: Vec<GlyphInstanceData>,
    pub masks: HashMap<GlyphKey, GlyphMaskData>,
}

/// Route for glyph output from [`draw_shaped_line`]: either CPU-blit into a pixel
/// canvas (existing paths — PDF, test) or collect instance + mask data for the GPU
/// atlas compositor (screen overlay).
pub(crate) enum GlyphSink<'a> {
    Canvas {
        canvas: &'a mut [u8],
        width: u32,
        height: u32,
    },
    Collect(&'a mut GlyphCollector),
}

/// Collect glyph instances and upright mask data for the GPU atlas compositor.
/// Blocks must be in canvas-texel coords (not tile-local); pen positions in the
/// returned [`GlyphCollector`] are canvas-texel coordinates ready for the GPU.
pub(crate) fn collect_overlay_glyphs(
    blocks: &[PreparedTextBlock],
    cache: &mut FontCache,
    fonts: &dyn FontProvider,
    opts: &RenderOptions,
) -> GlyphCollector {
    cache.clear_chains();
    let mut collector = GlyphCollector::default();
    for block in blocks {
        if block.translated_text.trim().is_empty() {
            continue;
        }
        let mut sink = GlyphSink::Collect(&mut collector);
        match block.layout_hints.layout_mode {
            OverlayLayoutMode::PerLine => render_per_line(&mut sink, block, cache, fonts, opts),
            OverlayLayoutMode::VerticalBlockRect => {
                render_vertical_block_rect(&mut sink, block, cache, fonts, opts)
            }
        }
    }
    collector
}

// ---------------------------------------------------------------------------
// Font cache

/// Per-render glyph timing/counters, accumulated thread-locally during a render and
/// read by the caller via [`take_render_stats`]. Instrumentation only — lets the
/// screen overlay attribute an `incr` present to shaping vs glyph rasterization vs
/// atlas reuse, so we can tell real raster work from a scheduling/memory stall.
#[derive(Clone, Copy, Default)]
pub struct RenderStats {
    pub shape_us: u64,
    pub raster_us: u64,
    pub glyphs_rastered: u32,
    pub glyphs_reused: u32,
}

thread_local! {
    static RENDER_STATS: std::cell::RefCell<RenderStats> = std::cell::RefCell::new(RenderStats::default());
}

/// Cap on the persistent glyph atlas; dropped+re-warmed past this. ~tens of MB at
/// worst (mask bytes scale with size²), generous for one language's glyphs × the
/// handful of px sizes the layout settles on.
const GLYPH_ATLAS_CAP: usize = 8192;

/// Take and reset the calling thread's accumulated render stats.
pub fn take_render_stats() -> RenderStats {
    RENDER_STATS.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

fn stat_shape(us: u64) {
    RENDER_STATS.with(|s| s.borrow_mut().shape_us += us);
}

fn stat_raster(us: u64) {
    RENDER_STATS.with(|s| {
        let mut st = s.borrow_mut();
        st.raster_us += us;
        st.glyphs_rastered += 1;
    });
}

fn stat_glyph_hit() {
    RENDER_STATS.with(|s| s.borrow_mut().glyphs_reused += 1);
}

/// One glyph's alpha-coverage mask, rasterized axis-aligned at the glyph origin
/// `(0, 0)`. `left`/`top` are the mask's offset from that origin (the side bearing /
/// ascent), so a draw blits it at `pen.round() + (left, top)`.
struct GlyphMask {
    cov: Vec<u8>,
    w: u32,
    h: u32,
    left: i32,
    top: i32,
}

#[derive(Default)]
pub(crate) struct FontCache {
    /// Parsed faces, keyed by handle. Each `Face` borrows the bytes held in
    /// `fonts`; declared first so it drops before the bytes it views. Caching the
    /// parsed face matters because `Face::from_slice` re-parses cmap + GSUB/GPOS,
    /// and a single block is shaped (whole + per-line) and rasterized — ~3 parses
    /// of the same font per run without this.
    faces: HashMap<FontHandle, Option<Face<'static>>>,
    fonts: HashMap<FontHandle, Option<Arc<Vec<u8>>>>,
    chains: HashMap<(Script, bool, bool, bool), Vec<FontHandle>>,
    /// Small dense id per font handle, so the glyph-atlas key is cheap ints.
    font_ids: HashMap<FontHandle, u32>,
    next_font_id: u32,
    /// Glyph alpha-mask atlas keyed by `(font id, glyph id, rounded px size)`. The
    /// same glyph recurs constantly within a render (every 'e', every space); this
    /// rasterizes each distinct one once and re-blits the mask. Keyed on the shaped
    /// *glyph id*, so ligatures / contextual forms are correct (they're distinct
    /// gids). `None` caches a no-outline glyph (space, .notdef) so it isn't retried.
    /// Axis-aligned only — a tilted line can't reuse an upright mask.
    glyphs: HashMap<(u32, u16, u32), Option<GlyphMask>>,
}

impl FontCache {
    /// Drop the font-chain lookups (which depend on the `FontProvider` — language,
    /// fallback config). Called at the start of each render so a reused cache can't
    /// serve a stale chain after a language change. The parsed faces and glyph atlas
    /// are font-file-keyed (provider-independent), so they persist — except the
    /// glyph atlas is dropped if it has grown past a cap, bounding memory over a
    /// long session (it re-warms on the next render).
    pub(crate) fn clear_chains(&mut self) {
        self.chains.clear();
        if self.glyphs.len() > GLYPH_ATLAS_CAP {
            self.glyphs.clear();
        }
    }

    /// Parse `handle` once and cache the `Face`, reused for shaping and rasterizing
    /// every run that uses this font in this render.
    fn face(&mut self, handle: &FontHandle) -> Option<&Face<'static>> {
        if !self.faces.contains_key(handle) {
            let parsed = self.bytes(handle).and_then(|arc| {
                // SAFETY: the same `Arc<Vec<u8>>` is held in `self.fonts` for the life
                // of this `FontCache` (`bytes` only inserts, never removes/mutates), so
                // its heap buffer outlives every `Face` stored here. The `'static`
                // lifetime is a lie scoped to the cache: faces are handed out only as
                // `&Face` borrowing `&self`, so none can outlive the bytes.
                let slice: &'static [u8] =
                    unsafe { core::mem::transmute::<&[u8], &'static [u8]>(arc.as_slice()) };
                Face::from_slice(slice, handle.ttc_index).map(|mut face| {
                    // Variable fonts (e.g. Android's Roboto: one file, `wght` axis)
                    // resolve every weight to the same file, so drive the axis to the
                    // handle's weight. A no-op on static faces — the picked file is
                    // already the right weight there. Sets it on the shared face, so
                    // shaping advances and glyph outlines both follow.
                    if handle.weight != 400 {
                        face.set_variations(&[rustybuzz::Variation {
                            tag: ttf_parser::Tag::from_bytes(b"wght"),
                            value: handle.weight as f32,
                        }]);
                    }
                    face
                })
            });
            self.faces.insert(handle.clone(), parsed);
        }
        self.faces.get(handle).and_then(|f| f.as_ref())
    }

    /// Stable small id for a font handle (resolved once per run, not per glyph).
    fn font_id(&mut self, handle: &FontHandle) -> u32 {
        if let Some(id) = self.font_ids.get(handle) {
            return *id;
        }
        let id = self.next_font_id;
        self.next_font_id += 1;
        self.font_ids.insert(handle.clone(), id);
        id
    }

    /// Atlas lookup: the alpha mask for glyph `gid` of font `fid`/`handle` at
    /// `size_px`. Rasterizes (axis-aligned at the glyph origin) on first miss, then
    /// reused. `scale = font_size / units_per_em`.
    fn glyph(
        &mut self,
        fid: u32,
        handle: &FontHandle,
        gid: u16,
        size_px: u32,
        scale: f32,
    ) -> Option<&GlyphMask> {
        let key = (fid, gid, size_px);
        if self.glyphs.contains_key(&key) {
            stat_glyph_hit();
        } else {
            let t = std::time::Instant::now();
            let gm = self
                .face(handle)
                .and_then(|face| raster_glyph(face, gid, scale));
            stat_raster(t.elapsed().as_micros() as u64);
            self.glyphs.insert(key, gm);
        }
        self.glyphs.get(&key).and_then(|g| g.as_ref())
    }

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
// Shaping
//
// BiDi segmentation + OpenType shaping live in [`crate::text_shape`], shared
// with the PDF overlay. This path adds the per-render font bookkeeping: it
// re-fetches the parsed `Face` from the `FontCache` and tags each shaped run
// with the `FontHandle` the rasterizer needs.

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ShapedRun {
    glyphs: Vec<ShapedGlyph>,
    units_per_em: i32,
    ascent: i32,
    descent: i32,
    /// Font this run was shaped with; the rasterizer re-fetches the parsed `Face`
    /// from the same per-render `FontCache`, so it's a cache hit rather than a parse.
    handle: FontHandle,
    rtl: bool,
    /// Byte offset of this run's first character in the source string
    /// passed to `shape_line`. Each glyph's `cluster` is relative to
    /// the run's slice; adding this gives the glyph's global byte
    /// position in the source — needed for the per-line breaker to
    /// compute exact prefix widths across an arbitrary source span.
    byte_start_in_text: usize,
}

fn shape_run(
    text: &str,
    run: &DirRun,
    handle: &FontHandle,
    cache: &mut FontCache,
) -> Option<ShapedRun> {
    let face = cache.face(handle)?;
    let shaped = text_shape::shape_run(text, run, face);
    Some(ShapedRun {
        glyphs: shaped.glyphs,
        units_per_em: shaped.units_per_em,
        ascent: shaped.ascent,
        descent: shaped.descent,
        handle: handle.clone(),
        rtl: shaped.rtl,
        byte_start_in_text: shaped.byte_start_in_text,
    })
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
    /// Cumulative em-advance indexed by source byte position. Length
    /// `source_text.len() + 1`; entry `i` is the total em-advance of
    /// every glyph whose source byte position is `< i`. Lets the
    /// line breaker compute exact widths of any source-text prefix
    /// without re-shaping. Mirrors what `FontMetrics::measure` does
    /// on the PDF path; the difference is we have the harfbuzz-
    /// shaped output here (with kerning, ligatures, complex script
    /// shaping) so the widths match what gets drawn pixel-for-pixel.
    cum_em_at_byte: Vec<f32>,
}

fn shape_line(
    text: &str,
    chain_lookup: &mut dyn FnMut(Script, &mut FontCache) -> Vec<FontHandle>,
    cache: &mut FontCache,
) -> LineShape {
    let t_shape = std::time::Instant::now();
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
    // Source-byte-ordered (byte_offset, em_advance) for every glyph
    // across every run. Used below to fold into a per-byte prefix sum.
    let mut glyph_advances: Vec<(usize, f32)> = Vec::new();
    for (_, run) in shaped {
        let upem = run.units_per_em as f32;
        let total_units: i64 = run.glyphs.iter().map(|g| g.advance_x as i64).sum();
        total_widths.push(total_units as f32 / upem);
        max_ascent_em = max_ascent_em.max(run.ascent as f32 / upem);
        max_descent_em = max_descent_em.max(-(run.descent as f32) / upem);
        let run_start = run.byte_start_in_text;
        for g in &run.glyphs {
            let global_byte = run_start + g.cluster as usize;
            glyph_advances.push((global_byte, g.advance_x as f32 / upem));
        }
        runs_only.push(run);
    }
    // Source order. For LTR latin text glyphs already arrive in source
    // order; this also makes the table correct for BiDi mixed runs
    // (visual order ≠ source order). Sort key is the source byte.
    glyph_advances.sort_by_key(|&(b, _)| b);
    let mut cum_em_at_byte = vec![0.0_f32; text.len() + 1];
    let mut acc = 0.0_f32;
    let mut g_idx = 0usize;
    for byte in 0..=text.len() {
        while g_idx < glyph_advances.len() && glyph_advances[g_idx].0 < byte {
            acc += glyph_advances[g_idx].1;
            g_idx += 1;
        }
        cum_em_at_byte[byte] = acc;
    }

    stat_shape(t_shape.elapsed().as_micros() as u64);
    LineShape {
        runs: runs_only,
        total_widths,
        max_ascent_em,
        max_descent_em,
        cum_em_at_byte,
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
    sink: &mut GlyphSink<'_>,
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
    let bold = block.is_bold;
    let mut size = block
        .layout_hints
        .suggested_font_size_px
        .max(opts.min_font_size_px);

    // Helper to shape the whole translated string for a given font size.
    // Shaping is size-independent (we shape once at upem and scale), so we
    // can reuse this across the size-shrink loop.
    let shaped_full = {
        let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
            c.chain_for(script, bold, false, false, &language, fonts)
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
            c.chain_for(script, bold, false, false, &language, fonts)
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
            sink,
            &line_shape,
            cache,
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
///
/// Width measurement uses the shaped per-glyph advances (via
/// `LineShape::cum_em_at_byte`) rather than an average-char-em
/// approximation. This is what the PDF text-erase path
/// (`wrap_lines_to_widths` in `pdf_overlay.rs`) does via
/// `FontMetrics::measure`. The previous approximation
/// underestimated prefixes dominated by wider-than-average glyphs —
/// e.g. capitalised first words on book / sign translations — which
/// made the breaker assign a prefix that "fit" the estimate but
/// overflowed the rect when actually drawn, clipping the trailing
/// glyph off-bitmap ("Ontwerpen Van" rendered as "Ontwerpen Va").
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

    // Width of `text[start_byte..end_byte]` at `font_size` using the
    // shaped per-glyph advances. Clamp byte indices to the table's
    // valid range — defensive, the breaker only passes positions
    // within bounds.
    let measure_bytes = |start_byte: usize, end_byte: usize| -> f32 {
        let lo = start_byte.min(shaped.cum_em_at_byte.len() - 1);
        let hi = end_byte.min(shaped.cum_em_at_byte.len() - 1);
        (shaped.cum_em_at_byte[hi] - shaped.cum_em_at_byte[lo]) * font_size
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
        if measure_bytes(cursor, bytes_text.len()) <= target_w {
            out.push(bytes_text[cursor..].to_string());
            cursor = bytes_text.len();
            continue;
        }
        if is_last {
            return None;
        }

        // Walk word boundaries (spaces) to find the largest prefix that fits.
        let remainder = &bytes_text[cursor..];
        let mut last_good_end: Option<usize> = None;
        for (off, ch) in remainder.char_indices() {
            if ch == ' ' {
                // Width of cursor..(cursor + off) — i.e. up to but not
                // including this space. The trailing space gets eaten by
                // the cursor advance below.
                if measure_bytes(cursor, cursor + off) <= target_w {
                    last_good_end = Some(off);
                } else {
                    break;
                }
            }
        }

        let Some(end) = last_good_end else {
            return None;
        };
        let assigned = &remainder[..end];
        out.push(assigned.to_string());
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
// Layout — VerticalBlockRect

/// Vertical (CJK top-to-bottom) blocks keep their vertical layout: the
/// translation is rendered rotated 90° CW, each line reading down the image,
/// successive lines advancing right-to-left — the same direction the source
/// columns flow. Layout happens in the transposed rect (wrap width = block
/// height, line stack capped by block width); the rotation maps it back.
fn render_vertical_block_rect(
    sink: &mut GlyphSink<'_>,
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
    let bold = block.is_bold;
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
        let candidate = wrap_into_block(translated, bh, size);
        let line_h = estimate_line_height(translated, size, &language, bold, cache, fonts);
        if line_h <= 0.0 {
            return;
        }
        let max_lines = (bw / line_h).floor() as usize;
        if candidate.len() <= max_lines.max(1) && all_lines_fit(&candidate, bh, size) {
            break candidate;
        }
        if size <= opts.min_font_size_px {
            return;
        }
        size -= 1.0;
    };

    let line_h = estimate_line_height(translated, size, &language, bold, cache, fonts);
    // Baseline offset within the transposed rect; in image space it advances
    // leftward from the block's right edge. 0.8 leaves the same ascender room
    // the horizontal block layout reserves below the top edge.
    let mut baseline_offset = line_h * 0.8;
    for line_text in lines {
        if line_text.trim().is_empty() {
            baseline_offset += line_h;
            continue;
        }
        let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
            c.chain_for(script, bold, false, false, &language, fonts)
                .to_vec()
        };
        let mut chain_fn = chain_fn;
        let line_shape = shape_line(&line_text, &mut chain_fn, cache);
        if line_shape.runs.is_empty() {
            baseline_offset += line_h;
            continue;
        }
        // Reading direction is image +y: rotation (cos, sin) = (0, 1).
        draw_shaped_line(
            sink,
            &line_shape,
            cache,
            block.bounding_box.right as f32 - baseline_offset,
            block.bounding_box.top as f32,
            0.0,
            1.0,
            size,
            block.foreground_argb,
        );
        baseline_offset += line_h;
    }
}

fn estimate_line_height(
    text: &str,
    size: f32,
    language: &str,
    bold: bool,
    cache: &mut FontCache,
    fonts: &dyn FontProvider,
) -> f32 {
    let chain_fn = |script: Script, c: &mut FontCache| -> Vec<FontHandle> {
        c.chain_for(script, bold, false, false, language, fonts)
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

#[allow(clippy::too_many_arguments)]
fn draw_shaped_line(
    sink: &mut GlyphSink<'_>,
    line: &LineShape,
    cache: &mut FontCache,
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
    //
    // Canvas path: axis-aligned lines reuse the glyph atlas (rasterized once, re-blitted per
    // pen position snapped to whole pixels); tilted lines rasterize per-glyph in line orientation.
    // Collect path: always uses the upright atlas entry and carries the line angle per-instance —
    // the GPU quad rotates for free, so tilted text reuses the same atlas entry as axis-aligned.
    let axis_aligned = sin_angle.abs() < 1e-3 && cos_angle > 0.0;
    let size_px = font_size.round().max(1.0) as u32;
    let mut cursor_x = 0.0f32;
    for run in &line.runs {
        let scale = font_size / run.units_per_em as f32;
        if axis_aligned {
            let fid = cache.font_id(&run.handle);
            for glyph in &run.glyphs {
                let glyph_x = origin_x + cursor_x + glyph.offset_x as f32 * scale;
                let glyph_y = origin_y - glyph.offset_y as f32 * scale;
                match sink {
                    GlyphSink::Canvas {
                        canvas,
                        width,
                        height,
                    } => {
                        if let Some(gm) = cache.glyph(fid, &run.handle, glyph.gid, size_px, scale) {
                            blit_mask(
                                canvas,
                                *width,
                                *height,
                                &gm.cov,
                                glyph_x.round() as i32 + gm.left,
                                glyph_y.round() as i32 + gm.top,
                                gm.w,
                                gm.h,
                                fg_argb,
                            );
                        }
                    }
                    GlyphSink::Collect(col) => {
                        let key = (fid, glyph.gid, size_px);
                        if let Some(gm) = cache.glyph(fid, &run.handle, glyph.gid, size_px, scale) {
                            let (cov, w, h, left, top) =
                                (gm.cov.clone(), gm.w, gm.h, gm.left, gm.top);
                            col.masks.entry(key).or_insert_with(|| GlyphMaskData {
                                key,
                                cov,
                                w,
                                h,
                                left,
                                top,
                            });
                            // ARGB → RGBA bytes (see the rotated branch below): the GPU
                            // glyph shader reads the colour as R,G,B,A, so `to_ne_bytes`
                            // would swap R/B for coloured ink.
                            let [a, r, g, b] = fg_argb.to_be_bytes();
                            col.instances.push(GlyphInstanceData {
                                key,
                                pen_x: glyph_x,
                                pen_y: glyph_y,
                                cos: cos_angle,
                                sin: sin_angle,
                                color: [r, g, b, a],
                            });
                        }
                    }
                }
                cursor_x += glyph.advance_x as f32 * scale;
            }
            continue;
        }
        // Non-axis-aligned: Canvas path rasterizes each glyph rotated in-line; Collect path
        // uses the upright atlas entry and carries the angle per-instance for the GPU to rotate.
        match sink {
            GlyphSink::Canvas {
                canvas,
                width,
                height,
            } => {
                let face = match cache.face(&run.handle) {
                    Some(f) => f,
                    None => continue,
                };
                for glyph in &run.glyphs {
                    let pen_local_x = cursor_x + glyph.offset_x as f32 * scale;
                    let pen_local_y = -(glyph.offset_y as f32) * scale;
                    let glyph_x = origin_x + pen_local_x * cos_angle - pen_local_y * sin_angle;
                    let glyph_y = origin_y + pen_local_x * sin_angle + pen_local_y * cos_angle;
                    let mut commands: Vec<Command> = Vec::new();
                    let mut outline_sink = OutlineSink {
                        builder: &mut commands,
                        origin_x: glyph_x,
                        origin_y: glyph_y,
                        scale,
                        cos_angle,
                        sin_angle,
                    };
                    if face
                        .outline_glyph(ttf_parser::GlyphId(glyph.gid), &mut outline_sink)
                        .is_some()
                    {
                        let (mask, placement) = Mask::new(commands.as_slice())
                            .format(Format::Alpha)
                            .render();
                        blit_mask(
                            canvas,
                            *width,
                            *height,
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
            GlyphSink::Collect(col) => {
                let fid = cache.font_id(&run.handle);
                for glyph in &run.glyphs {
                    let pen_local_x = cursor_x + glyph.offset_x as f32 * scale;
                    let pen_local_y = -(glyph.offset_y as f32) * scale;
                    let glyph_x = origin_x + pen_local_x * cos_angle - pen_local_y * sin_angle;
                    let glyph_y = origin_y + pen_local_x * sin_angle + pen_local_y * cos_angle;
                    let key = (fid, glyph.gid, size_px);
                    if let Some(gm) = cache.glyph(fid, &run.handle, glyph.gid, size_px, scale) {
                        let (cov, w, h, left, top) = (gm.cov.clone(), gm.w, gm.h, gm.left, gm.top);
                        col.masks.entry(key).or_insert_with(|| GlyphMaskData {
                            key,
                            cov,
                            w,
                            h,
                            left,
                            top,
                        });
                        // `fg_argb` is ARGB; the GPU glyph shader reads the instance
                        // colour bytes as R,G,B,A, so pack RGBA explicitly. (`to_ne_bytes`
                        // would emit B,G,R,A on little-endian — invisible for white text,
                        // but it swaps R/B for coloured ink, turning navy into brown.)
                        let [a, r, g, b] = fg_argb.to_be_bytes();
                        col.instances.push(GlyphInstanceData {
                            key,
                            pen_x: glyph_x,
                            pen_y: glyph_y,
                            cos: cos_angle,
                            sin: sin_angle,
                            color: [r, g, b, a],
                        });
                    }
                    cursor_x += glyph.advance_x as f32 * scale;
                }
            }
        }
    }
}

/// Rasterize one glyph's alpha mask axis-aligned at the origin `(0, 0)` and `scale`
/// — the atlas entry, re-blitted at each pen position. Returns `None` for glyphs
/// with no outline (space, .notdef).
fn raster_glyph(face: &Face, gid: u16, scale: f32) -> Option<GlyphMask> {
    let mut commands: Vec<Command> = Vec::new();
    let mut sink = OutlineSink {
        builder: &mut commands,
        origin_x: 0.0,
        origin_y: 0.0,
        scale,
        cos_angle: 1.0,
        sin_angle: 0.0,
    };
    face.outline_glyph(ttf_parser::GlyphId(gid), &mut sink)?;
    if commands.is_empty() {
        return None;
    }
    let (mask, placement) = Mask::new(commands.as_slice())
        .format(Format::Alpha)
        .render();
    if placement.width == 0 || placement.height == 0 {
        return None;
    }
    Some(GlyphMask {
        cov: mask,
        w: placement.width,
        h: placement.height,
        left: placement.left,
        top: placement.top,
    })
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
    // Match the rest of this canvas, which `crate::ocr` writes (erase, fill,
    // sample) with `argb.to_ne_bytes()`. The glyph colour must use the same
    // little-endian byte order so it isn't R/B-swapped relative to the
    // background it's drawn over.
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
            // Source-over alpha. Previously this loop only blended
            // RGB and left the alpha channel untouched, which worked
            // when the canvas was pre-filled with an opaque bg
            // rectangle but produced fully-transparent glyph pixels
            // (RGB set, A=0) when painted onto a transparent canvas
            // (the text-only layer of the bg/text split).
            let dst_a = canvas[buf_idx + 3] as f32 / 255.0;
            let out_a = a + dst_a * inv;
            canvas[buf_idx + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
}

//! Writeback: take translated pages and the original PDF bytes, emit a new
//! PDF where the original text has been **surgically removed** from the
//! content stream and the translated text drawn in its place.
//!
//! The surgery walks the page's content stream operators tracking the text
//! matrix (Tm/Td/TD/T*), graphics state stack (q/Q), and CTM (cm). For each
//! text-show operator (Tj/TJ/'/"), we compute the origin point in PDF user
//! space; if it falls inside any translated block's bbox, we drop the op.
//! Once originals are gone there's no overpaint hack — we just append a
//! translated-text-only stream.
//!
//! v1 limitations (intentional):
//! - Latin output only. Uses Helvetica from the PDF standard 14 (no font
//!   embedding). Cross-script (CJK / Arabic / etc.) will become `?`.
//! - Approximate Helvetica width metrics (~0.5em per char).
//! - Black text only; no per-source-style coloring.

use std::io::Write as _;

use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};

use crate::ocr::Rect;
use crate::pdf::{PageDims, PdfError};
use crate::pdf_translate::PageTranslationResult;
use crate::styled::TranslatedStyledBlock;

/// PDF resource names for our font variants. All eight are PDF standard-14
/// base fonts, so no embedding is needed.
const HELVETICA_REGULAR: &[u8] = b"TrHelv";
const HELVETICA_BOLD: &[u8] = b"TrHelvB";
const HELVETICA_OBLIQUE: &[u8] = b"TrHelvI";
const HELVETICA_BOLD_OBLIQUE: &[u8] = b"TrHelvBI";
const COURIER_REGULAR: &[u8] = b"TrCour";
const COURIER_BOLD: &[u8] = b"TrCourB";
const COURIER_OBLIQUE: &[u8] = b"TrCourI";
const COURIER_BOLD_OBLIQUE: &[u8] = b"TrCourBI";

/// Style sampled from the original Tjs that fell inside a removal rect.
/// Used to pick a font variant, a fill color, a target font size, and a
/// baseline anchor for the translation.
#[derive(Debug, Clone)]
struct SampledBlockStyle {
    bold: bool,
    italic: bool,
    monospace: bool,
    /// Non-stroking fill colour as RGB in `[0, 1]`.
    fill_rgb: (f32, f32, f32),
    /// Median font size (in points) of the dropped Tjs, or `None` if no
    /// Tjs were dropped for this rect.
    font_size: Option<f32>,
    /// User-space baseline-leading-edge anchor of the visually-top-left line
    /// among the dropped Tjs. Used to place the first translated line at the
    /// same baseline as the original.
    anchor: Option<(f32, f32)>,
    /// Number of distinct visual baselines across the dropped Tjs — i.e.
    /// how many lines the original text actually occupied. More reliable
    /// than deriving from bbox height.
    original_line_count: usize,
    /// `(a, b, c, d)` of the original Tj's combined `text_matrix × CTM`.
    /// We reuse it for our emitted glyphs so the new text inherits the
    /// producer's orientation (essential for /Rotate-compensating producers
    /// and Y-flip top-left-origin streams). Defaults to identity (no
    /// orientation change) when no samples are available.
    text_orientation: (f32, f32, f32, f32),
    /// Visually top-to-bottom baseline starts sampled from the original text.
    /// These preserve hanging indents where continuation lines start further
    /// left than the first line.
    line_anchors: Vec<(f32, f32)>,
}

impl Default for SampledBlockStyle {
    fn default() -> Self {
        Self {
            bold: false,
            italic: false,
            monospace: false,
            fill_rgb: (0.0, 0.0, 0.0),
            font_size: None,
            anchor: None,
            original_line_count: 0,
            text_orientation: (1.0, 0.0, 0.0, 1.0),
            line_anchors: Vec::new(),
        }
    }
}

impl SampledBlockStyle {
    fn font_resource(&self) -> &'static [u8] {
        match (self.monospace, self.bold, self.italic) {
            (true, true, true) => COURIER_BOLD_OBLIQUE,
            (true, true, false) => COURIER_BOLD,
            (true, false, true) => COURIER_OBLIQUE,
            (true, false, false) => COURIER_REGULAR,
            (false, true, true) => HELVETICA_BOLD_OBLIQUE,
            (false, true, false) => HELVETICA_BOLD,
            (false, false, true) => HELVETICA_OBLIQUE,
            (false, false, false) => HELVETICA_REGULAR,
        }
    }
}

/// Approximate average Helvetica glyph width as a fraction of font size.
const HELVETICA_AVG_ADVANCE: f32 = 0.5;

/// Vertical margin inside the bbox so descenders don't clip the bottom.
const TEXT_BASELINE_PAD: f32 = 0.2;

/// Leading multiplier for wrapped lines (line-height = font_size * factor).
const LINE_HEIGHT_FACTOR: f32 = 1.15;

#[derive(Debug)]
pub enum PdfWriteError {
    Lopdf(lopdf::Error),
    Pdf(PdfError),
    Io(std::io::Error),
    Other(String),
}

impl From<lopdf::Error> for PdfWriteError {
    fn from(value: lopdf::Error) -> Self {
        Self::Lopdf(value)
    }
}

impl From<PdfError> for PdfWriteError {
    fn from(value: PdfError) -> Self {
        Self::Pdf(value)
    }
}

impl From<std::io::Error> for PdfWriteError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl std::fmt::Display for PdfWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lopdf(err) => write!(f, "lopdf: {err}"),
            Self::Pdf(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "io: {err}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for PdfWriteError {}

/// Build a translated PDF by removing the original text from each page and
/// appending the translated text in the same bbox positions.
///
/// `fonts` is consulted for non-Standard-14 scripts and for accurate wrap
/// metrics. Pass [`crate::font_provider::NoFontProvider`] (or `&|_| None`) to
/// keep the current Standard-14-only behavior.
pub fn write_translated_pdf(
    original_pdf_bytes: &[u8],
    translations: &[PageTranslationResult],
    fonts: &dyn crate::font_provider::FontProvider,
) -> Result<Vec<u8>, PdfWriteError> {
    use crate::font_provider::{FontHandle, FontRequest};
    type FontKey = (FontRequest, FontHandle);

    let mut doc = Document::load_mem(original_pdf_bytes)?;

    let pages: Vec<(u32, ObjectId)> = doc.get_pages().into_iter().collect();

    // ----- Pass 1: surgery + style sampling per page. We need styles before
    // we can build font requests, so do all surgery first; defer overlays.
    struct PageWork<'a> {
        page_id: ObjectId,
        translation: &'a PageTranslationResult,
        removal_rects: Vec<UserRect>,
        block_styles: Vec<SampledBlockStyle>,
        final_ctm: Matrix,
        geom: PageGeometry,
    }

    let mut works: Vec<PageWork> = Vec::new();
    let mut modified_pages = std::collections::HashSet::new();
    for translation in translations {
        let Some((_, page_id)) = pages
            .iter()
            .find(|(num, _)| (*num as usize).saturating_sub(1) == translation.page_index)
        else {
            return Err(PdfWriteError::Other(format!(
                "translation refers to page index {} which does not exist",
                translation.page_index
            )));
        };
        if translation.blocks.is_empty() {
            continue;
        }
        let geom = PageGeometry::read(&doc, *page_id, translation.page);
        let removal_rects: Vec<UserRect> = translation
            .blocks
            .iter()
            .map(|b| user_rect_from_display(b.bounding_box, geom))
            .collect();
        let (final_ctm, block_styles) =
            rewrite_page_content(&mut doc, *page_id, &removal_rects, geom)?;
        ensure_fonts_in_page_resources(&mut doc, *page_id)?;
        modified_pages.insert(*page_id);
        works.push(PageWork {
            page_id: *page_id,
            translation,
            removal_rects,
            block_styles,
            final_ctm,
            geom,
        });
    }

    // ----- Pass 2: walk every block, group by (FontRequest, FontHandle) and
    // accumulate the union of every char that font needs to render. Each
    // block contributes a request for its dominant style **plus** one per
    // distinct (bold, italic) variant in its style_spans, so intra-block
    // bold / italic words have a font ready when we emit them.
    let mut union_text: std::collections::HashMap<FontKey, String> =
        std::collections::HashMap::new();
    for work in &works {
        for (block, style) in work.translation.blocks.iter().zip(work.block_styles.iter()) {
            let mut variants: std::collections::HashSet<(bool, bool)> =
                std::collections::HashSet::new();
            variants.insert((style.bold, style.italic));
            for span in &block.style_spans {
                if let Some(s) = &span.style {
                    variants.insert((s.bold, s.italic));
                }
            }
            for (bold, italic) in variants {
                let req = FontRequest {
                    language: work.translation.target_language.clone(),
                    bold,
                    italic,
                    monospace: style.monospace,
                };
                if let Some(handle) = fonts.locate(&req) {
                    union_text
                        .entry((req, handle))
                        .or_default()
                        .push_str(&block.text);
                }
            }
        }
    }

    // ----- Pass 3: parse each unique font once with its document-wide union
    // text so the subset embedded later covers every page that uses it.
    let mut metrics_cache: std::collections::HashMap<FontKey, crate::font_metrics::FontMetrics> =
        std::collections::HashMap::new();
    for (key, text) in &union_text {
        let (_req, handle) = key;
        match crate::font_metrics::FontMetrics::from_file_for_text(
            &handle.path,
            handle.ttc_index,
            text,
        ) {
            Ok(m) => {
                metrics_cache.insert(key.clone(), m);
            }
            Err(e) => {
                eprintln!(
                    "[pdf_write] could not parse {} (ttc_index={}): {e}",
                    handle.path.display(),
                    handle.ttc_index,
                );
            }
        }
    }

    // ----- Pass 4: embed each unique font once. Single subset per font is
    // shared across every page that references it.
    let mut embed_cache: std::collections::HashMap<FontKey, crate::pdf_font_embed::EmbeddedFont> =
        std::collections::HashMap::new();
    let mut next_slot = 0usize;
    for (key, metrics) in &metrics_cache {
        if let Some(e) = crate::pdf_font_embed::embed_font(&mut doc, metrics, next_slot) {
            embed_cache.insert(key.clone(), e);
            next_slot += 1;
        }
    }

    // ----- Pass 5: per-page emit overlay using cached metrics + embeds.
    let helvetica_fallback = crate::font_metrics::FontMetrics::approx(HELVETICA_AVG_ADVANCE);
    let courier_fallback = crate::font_metrics::FontMetrics::approx(0.6);
    for work in &works {
        // Per block: a small map `(bold, italic) -> (FontMetrics, Option<Embed>)`
        // covering every variant seen in style_spans plus the dominant.
        let mut block_resources: Vec<BlockResources> =
            Vec::with_capacity(work.translation.blocks.len());
        for (block, style) in work.translation.blocks.iter().zip(work.block_styles.iter()) {
            let mut variants: std::collections::HashSet<(bool, bool)> =
                std::collections::HashSet::new();
            variants.insert((style.bold, style.italic));
            for span in &block.style_spans {
                if let Some(s) = &span.style {
                    variants.insert((s.bold, s.italic));
                }
            }
            let mut by_flags: std::collections::HashMap<
                (bool, bool),
                (
                    crate::font_metrics::FontMetrics,
                    Option<crate::pdf_font_embed::EmbeddedFont>,
                ),
            > = std::collections::HashMap::new();
            for (bold, italic) in variants {
                let req = FontRequest {
                    language: work.translation.target_language.clone(),
                    bold,
                    italic,
                    monospace: style.monospace,
                };
                let key = fonts.locate(&req).map(|handle| (req, handle));
                let metrics = key
                    .as_ref()
                    .and_then(|k| metrics_cache.get(k).cloned())
                    .unwrap_or_else(|| {
                        if style.monospace {
                            courier_fallback.clone()
                        } else {
                            helvetica_fallback.clone()
                        }
                    });
                let embed = key.as_ref().and_then(|k| embed_cache.get(k).cloned());
                by_flags.insert((bold, italic), (metrics, embed));
            }
            block_resources.push(BlockResources {
                by_flags,
                default_flags: (style.bold, style.italic),
                monospace: style.monospace,
            });
        }
        let block_embeds: Vec<Option<crate::pdf_font_embed::EmbeddedFont>> = block_resources
            .iter()
            .flat_map(|r| r.by_flags.values().map(|(_, e)| e.clone()))
            .collect();
        attach_embedded_fonts_to_page(&mut doc, work.page_id, &block_embeds)?;
        let overlay_stream = build_overlay_stream(
            &work.translation.blocks,
            &work.removal_rects,
            &work.block_styles,
            &block_resources,
            work.geom,
            work.final_ctm,
        );
        append_content_stream(&mut doc, work.page_id, overlay_stream)?;
    }

    prune_unused_fonts(&mut doc, &modified_pages)?;

    let mut out = Vec::new();
    doc.save_to(&mut out)?;
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct UserRect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl UserRect {
    fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }
}

/// User-space dimensions read from the page's `/MediaBox`, honouring `/Rotate`.
#[derive(Debug, Clone, Copy)]
struct PageGeometry {
    /// User-space width (independent of `/Rotate`; matches MediaBox x range).
    user_w: f32,
    /// User-space height (independent of `/Rotate`; matches MediaBox y range).
    user_h: f32,
    /// `/Rotate` value, normalised to 0/90/180/270.
    rotate: i32,
}

impl PageGeometry {
    fn read(doc: &Document, page_id: ObjectId, fallback_display: PageDims) -> Self {
        let page = doc.get_object(page_id).and_then(Object::as_dict);
        let rotate = page
            .as_ref()
            .ok()
            .and_then(|p| p.get(b"Rotate").ok())
            .and_then(|o| o.as_i64().ok())
            .unwrap_or(0);
        let rotate = ((rotate % 360 + 360) % 360) as i32;

        // mupdf's `bounds()` returns post-rotation display dims. Convert back
        // to MediaBox-relative user dims so coordinate math here aligns with
        // PDF user space.
        let (user_w, user_h) = match rotate {
            90 | 270 => (fallback_display.height_pts, fallback_display.width_pts),
            _ => (fallback_display.width_pts, fallback_display.height_pts),
        };

        // Prefer the actual MediaBox if present (handles non-zero origins).
        if let Ok(p) = page {
            if let Ok(Object::Array(arr)) = p.get(b"MediaBox") {
                if arr.len() == 4 {
                    let nums: Option<Vec<f32>> = arr.iter().map(object_as_f32).collect();
                    if let Some(n) = nums {
                        return Self {
                            user_w: n[2] - n[0],
                            user_h: n[3] - n[1],
                            rotate,
                        };
                    }
                }
            }
        }

        Self {
            user_w,
            user_h,
            rotate,
        }
    }
}

/// Convert a mupdf stext bbox (display coords, top-left origin) to a PDF
/// user-space rect honouring the page's `/Rotate` attribute.
fn user_rect_from_display(bbox: Rect, geom: PageGeometry) -> UserRect {
    let (l, t, r, b) = (
        bbox.left as f32,
        bbox.top as f32,
        bbox.right as f32,
        bbox.bottom as f32,
    );
    // Inverse of the user→display rotation. Display has top-left origin
    // (y down); user space has bottom-left origin (y up).
    //   R=0:    Ux=Dx,           Uy=H-Dy
    //   R=90:   Ux=Dy,           Uy=Dx
    //   R=180:  Ux=W-Dx,         Uy=Dy
    //   R=270:  Ux=W-Dy,         Uy=H-Dx
    let (x0, x1, y0, y1) = match geom.rotate {
        0 => (l, r, geom.user_h - b, geom.user_h - t),
        90 => (t, b, l, r),
        180 => (geom.user_w - r, geom.user_w - l, t, b),
        270 => (
            geom.user_w - b,
            geom.user_w - t,
            geom.user_h - r,
            geom.user_h - l,
        ),
        _ => (l, r, geom.user_h - b, geom.user_h - t),
    };
    UserRect {
        x0: x0.min(x1),
        x1: x0.max(x1),
        y0: y0.min(y1),
        y1: y0.max(y1),
    }
}

// ---------------------------------------------------------------------------
// Content-stream surgery.
// ---------------------------------------------------------------------------

/// Walk the page's decoded content stream, drop every text-show operator
/// whose origin lies inside any of `removal_rects`, and write the result
/// back. Non-text operators (paths, images, shading) are left untouched.
/// Returns the CTM that's still active at the end of the content stream
/// so the appended translated-text stream can match the producer's local
/// coordinate convention.
fn rewrite_page_content(
    doc: &mut Document,
    page_id: ObjectId,
    removal_rects: &[UserRect],
    geom: PageGeometry,
) -> Result<(Matrix, Vec<SampledBlockStyle>), PdfWriteError> {
    let content = doc.get_and_decode_page_content(page_id)?;
    let font_advances = FontAdvanceMap::from_page(doc, page_id);
    let (filtered, final_ctm, raw_samples) =
        filter_text_ops(content.operations, removal_rects, &font_advances);
    let new_bytes = Content {
        operations: filtered,
    }
    .encode()?;
    let block_styles = resolve_block_styles(doc, page_id, &raw_samples, geom);
    doc.change_page_content(page_id, new_bytes)?;
    Ok((final_ctm, block_styles))
}

/// For each removal rect, pick the dominant raw style sample (most common
/// font + the median fill colour) and resolve its font resource to a
/// `SampledBlockStyle` with bold/italic flags.
fn resolve_block_styles(
    doc: &Document,
    page_id: ObjectId,
    raw: &[Vec<RawStyleSample>],
    geom: PageGeometry,
) -> Vec<SampledBlockStyle> {
    let fonts = doc.get_page_fonts(page_id).ok();
    raw.iter()
        .map(|samples| {
            if samples.is_empty() {
                return SampledBlockStyle::default();
            }

            // Dominant font resource = mode of font_resource values.
            let mut font_counts: std::collections::HashMap<Vec<u8>, usize> =
                std::collections::HashMap::new();
            for sample in samples {
                if let Some(name) = &sample.font_resource {
                    *font_counts.entry(name.clone()).or_default() += 1;
                }
            }
            let dominant_font = font_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(name, _)| name);

            // Mean fill color across samples.
            let n = samples.len() as f32;
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            for s in samples {
                r += s.fill_rgb.0;
                g += s.fill_rgb.1;
                b += s.fill_rgb.2;
            }
            let fill_rgb = (r / n, g / n, b / n);

            let flags = dominant_font.as_deref().and_then(|res_name| {
                let fonts = fonts.as_ref()?;
                let font_dict = fonts.get(res_name)?;
                Some(font_flags(doc, font_dict))
            });
            let (bold, italic, monospace) = flags.unwrap_or_default();

            // Median font size across the rect's samples. Median (not mean)
            // is robust against the occasional outlier Tj that doesn't
            // belong to the main run of text in this block.
            let mut sizes: Vec<f32> = samples.iter().map(|s| s.font_size).collect();
            sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let font_size = if sizes.is_empty() {
                None
            } else {
                Some(sizes[sizes.len() / 2])
            };

            // Anchor = the visually top-left baseline among the sampled Tjs.
            // We sort by visual y first (smallest visual_y = visually
            // topmost line), then by visual x (leftmost on that line).
            let anchor = samples
                .iter()
                .min_by(|a, b| {
                    let (ax, ay) = visual_position(a.origin, geom);
                    let (bx, by) = visual_position(b.origin, geom);
                    ay.partial_cmp(&by)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
                })
                .map(|s| s.origin);

            let line_anchors = original_line_anchors(samples, geom);
            let original_line_count = line_anchors.len().max(1);

            // Pick the orientation of the first sample. Producers almost
            // always use one orientation per block; if it's mixed, the first
            // is as good as any and the others would visually clash anyway.
            let text_orientation = samples
                .first()
                .map(|s| s.text_orientation)
                .unwrap_or((1.0, 0.0, 0.0, 1.0));

            SampledBlockStyle {
                bold,
                italic,
                monospace,
                fill_rgb,
                font_size,
                anchor,
                original_line_count,
                text_orientation,
                line_anchors,
            }
        })
        .collect()
}

fn original_line_anchors(samples: &[RawStyleSample], geom: PageGeometry) -> Vec<(f32, f32)> {
    let mut positioned: Vec<(f32, f32, (f32, f32))> = samples
        .iter()
        .map(|s| {
            let (vx, vy) = visual_position(s.origin, geom);
            (vy, vx, s.origin)
        })
        .collect();
    positioned.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut anchors = Vec::new();
    let mut current_y: Option<f32> = None;
    let mut best_x = f32::INFINITY;
    let mut best_origin = (0.0, 0.0);
    for (vy, vx, origin) in positioned {
        if current_y.is_some_and(|y| (vy - y).abs() >= 1.0) {
            anchors.push(best_origin);
            current_y = Some(vy);
            best_x = vx;
            best_origin = origin;
            continue;
        }
        if current_y.is_none() {
            current_y = Some(vy);
        }
        if vx < best_x {
            best_x = vx;
            best_origin = origin;
        }
    }
    if current_y.is_some() {
        anchors.push(best_origin);
    }
    anchors
}

/// Map a PDF user-space point to a "visual" coordinate where smaller is
/// more visually top-left. Used to pick the anchor among multiple Tjs.
fn visual_position(user: (f32, f32), geom: PageGeometry) -> (f32, f32) {
    // Forward mapping user → display (smaller display y/x = visually top/left).
    //   R=0:    Dx=Ux,           Dy=H-Uy
    //   R=90:   Dx=Uy,           Dy=Ux
    //   R=180:  Dx=W-Ux,         Dy=Uy
    //   R=270:  Dx=H-Uy,         Dy=W-Ux
    match geom.rotate {
        0 => (user.0, geom.user_h - user.1),
        90 => (user.1, user.0),
        180 => (geom.user_w - user.0, user.1),
        270 => (geom.user_h - user.1, geom.user_w - user.0),
        _ => (user.0, geom.user_h - user.1),
    }
}

/// Detect (bold, italic, monospace) for a font.
///
/// Monospace is read from the FontDescriptor `/Flags` bit 1 (`FixedPitch`)
/// when a descriptor is present; otherwise we fall back to name patterns
/// (the PDF standard-14 Couriers don't ship a descriptor). Bold/italic come
/// from the `/Flags` `ForceBold`/`Italic` bits when present, with name
/// patterns as a fallback.
fn font_flags(doc: &Document, font_dict: &Dictionary) -> (bool, bool, bool) {
    let mut flags_int: Option<i64> = None;
    if let Ok(descriptor_ref) = font_dict.get(b"FontDescriptor") {
        let descriptor = match descriptor_ref {
            Object::Reference(id) => doc.get_dictionary(*id).ok(),
            Object::Dictionary(d) => Some(d),
            _ => None,
        };
        if let Some(d) = descriptor {
            flags_int = d.get(b"Flags").ok().and_then(|o| o.as_i64().ok());
        }
    }

    // Per PDF spec table 123:
    //   bit 1  = FixedPitch (1 << 0)
    //   bit 7  = Italic     (1 << 6)
    //   bit 19 = ForceBold  (1 << 18)
    let mut monospace = flags_int.map(|f| f & (1 << 0) != 0).unwrap_or(false);
    let italic_flag = flags_int.map(|f| f & (1 << 6) != 0).unwrap_or(false);
    let bold_flag = flags_int.map(|f| f & (1 << 18) != 0).unwrap_or(false);

    let base_font = font_dict
        .get(b"BaseFont")
        .ok()
        .and_then(|o| o.as_name().ok())
        .unwrap_or(b"");
    let (name_bold, name_italic, name_monospace) = detect_from_name(base_font);

    monospace = monospace || name_monospace;
    let bold = bold_flag || name_bold;
    let italic = italic_flag || name_italic;
    (bold, italic, monospace)
}

/// Heuristic name-pattern detection for bold/italic/monospace, used as a
/// fallback when the FontDescriptor flags aren't available (e.g. standard
/// 14 fonts) or to catch producers that don't set the flags.
fn detect_from_name(base_font: &[u8]) -> (bool, bool, bool) {
    // Strip the optional 6-letter subset tag prefix (`AAAAAA+ArialMT`).
    let name = match base_font.iter().position(|&b| b == b'+') {
        Some(idx) if idx == 6 => &base_font[idx + 1..],
        _ => base_font,
    };
    let lower: Vec<u8> = name.iter().map(|b| b.to_ascii_lowercase()).collect();
    let lower_str = std::str::from_utf8(&lower).unwrap_or("");
    let bold = ["bold", "heavy", "black", "semibold", "demibold"]
        .iter()
        .any(|kw| lower_str.contains(kw));
    let italic = lower_str.contains("italic") || lower_str.contains("oblique");
    let monospace = [
        "courier",
        "mono",
        "consolas",
        "menlo",
        "monaco",
        "inconsolata",
        "sourcecodepro",
        "firacode",
        "jetbrainsmono",
        "robotomono",
        "hack",
        "fixedsys",
        "lucidaconsole",
    ]
    .iter()
    .any(|kw| lower_str.contains(kw));
    (bold, italic, monospace)
}

#[derive(Debug, Clone)]
struct FontAdvance {
    code_bytes: usize,
    default_width: f32,
    widths: std::collections::HashMap<u16, f32>,
}

impl Default for FontAdvance {
    fn default() -> Self {
        Self {
            code_bytes: 1,
            default_width: 500.0,
            widths: std::collections::HashMap::new(),
        }
    }
}

impl FontAdvance {
    fn from_font_dict(doc: &Document, font: &Dictionary) -> Self {
        let subtype = font
            .get(b"Subtype")
            .ok()
            .and_then(|o| o.as_name().ok())
            .unwrap_or(b"");
        if subtype == b"Type0" {
            return Self::from_type0(doc, font);
        }
        Self::from_simple_font(font)
    }

    fn from_simple_font(font: &Dictionary) -> Self {
        let first = font
            .get(b"FirstChar")
            .ok()
            .and_then(|o| o.as_i64().ok())
            .unwrap_or(0)
            .max(0) as u16;
        let mut widths = std::collections::HashMap::new();
        if let Ok(Object::Array(arr)) = font.get(b"Widths") {
            for (i, width) in arr.iter().enumerate() {
                if let Some(w) = object_as_f32(width) {
                    widths.insert(first.saturating_add(i as u16), w);
                }
            }
        }
        Self {
            code_bytes: 1,
            default_width: 500.0,
            widths,
        }
    }

    fn from_type0(doc: &Document, font: &Dictionary) -> Self {
        let descendant = font
            .get(b"DescendantFonts")
            .ok()
            .and_then(|o| o.as_array().ok())
            .and_then(|arr| arr.first())
            .and_then(|obj| match obj {
                Object::Reference(id) => doc.get_dictionary(*id).ok(),
                Object::Dictionary(d) => Some(d),
                _ => None,
            });
        let Some(descendant) = descendant else {
            return Self {
                code_bytes: 2,
                default_width: 1000.0,
                widths: std::collections::HashMap::new(),
            };
        };
        let default_width = descendant
            .get(b"DW")
            .ok()
            .and_then(object_as_f32)
            .unwrap_or(1000.0);
        let mut out = Self {
            code_bytes: 2,
            default_width,
            widths: std::collections::HashMap::new(),
        };
        if let Ok(Object::Array(w_array)) = descendant.get(b"W") {
            parse_cid_widths(w_array, &mut out.widths);
        }
        out
    }

    fn width_for_code(&self, code: u16) -> f32 {
        self.widths
            .get(&code)
            .copied()
            .unwrap_or(self.default_width)
    }

    fn string_width_1000(&self, bytes: &[u8]) -> (f32, usize, usize) {
        let mut width = 0.0;
        let mut glyphs = 0usize;
        let mut spaces = 0usize;
        if self.code_bytes == 2 {
            for chunk in bytes.chunks(2) {
                let code = if chunk.len() == 2 {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                } else {
                    chunk[0] as u16
                };
                width += self.width_for_code(code);
                glyphs += 1;
                if code == 32 {
                    spaces += 1;
                }
            }
        } else {
            for &b in bytes {
                width += self.width_for_code(b as u16);
                glyphs += 1;
                if b == b' ' {
                    spaces += 1;
                }
            }
        }
        (width, glyphs, spaces)
    }
}

#[derive(Debug, Default, Clone)]
struct FontAdvanceMap {
    by_resource: std::collections::HashMap<Vec<u8>, FontAdvance>,
}

impl FontAdvanceMap {
    fn from_page(doc: &Document, page_id: ObjectId) -> Self {
        let mut by_resource = std::collections::HashMap::new();
        if let Ok(fonts) = doc.get_page_fonts(page_id) {
            for (name, font) in fonts {
                by_resource.insert(name, FontAdvance::from_font_dict(doc, font));
            }
        }
        Self { by_resource }
    }

    fn get(&self, name: Option<&Vec<u8>>) -> FontAdvance {
        name.and_then(|n| self.by_resource.get(n))
            .cloned()
            .unwrap_or_default()
    }
}

fn parse_cid_widths(w_array: &[Object], widths: &mut std::collections::HashMap<u16, f32>) {
    let mut i = 0usize;
    while i < w_array.len() {
        let Some(first) = w_array
            .get(i)
            .and_then(object_as_f32)
            .map(|v| v.max(0.0) as u16)
        else {
            i += 1;
            continue;
        };
        let Some(next) = w_array.get(i + 1) else {
            break;
        };
        match next {
            Object::Array(arr) => {
                for (offset, width) in arr.iter().enumerate() {
                    if let Some(w) = object_as_f32(width) {
                        widths.insert(first.saturating_add(offset as u16), w);
                    }
                }
                i += 2;
            }
            _ => {
                if let (Some(last), Some(width)) = (
                    w_array.get(i + 1).and_then(object_as_f32),
                    w_array.get(i + 2).and_then(object_as_f32),
                ) {
                    for code in first..=(last.max(first as f32) as u16) {
                        widths.insert(code, width);
                    }
                    i += 3;
                } else {
                    i += 1;
                }
            }
        }
    }
}

/// One observation of original-text style for a single removal rect.
#[derive(Debug, Clone)]
struct RawStyleSample {
    font_resource: Option<Vec<u8>>,
    fill_rgb: (f32, f32, f32),
    font_size: f32,
    /// User-space origin of the Tj — i.e. the baseline-leading-edge of the
    /// original text, after CTM has been applied.
    origin: (f32, f32),
    /// `(a, b, c, d)` of the *combined* `text_matrix × CTM` at this Tj —
    /// captures whatever rotation/flip the producer applied. Used to keep
    /// our emitted glyphs in the same orientation. Pure translation
    /// originals come out as `(1, 0, 0, 1)`, Y-flip producers as
    /// `(1, 0, 0, -1)`, /Rotate-compensating producers as `(0, 1, -1, 0)`.
    text_orientation: (f32, f32, f32, f32),
}

/// Track text/graphics state across `ops` and drop any text-show op whose
/// current origin lies inside a removal rect. Returns the filtered op list,
/// the CTM still active after the (balanced) graphics-state stack has
/// unwound, and per-rect samples of the dropped Tjs' font + fill colour so
/// the appended translation can mimic the producer's style.
fn filter_text_ops(
    ops: Vec<Operation>,
    removal_rects: &[UserRect],
    font_advances: &FontAdvanceMap,
) -> (Vec<Operation>, Matrix, Vec<Vec<RawStyleSample>>) {
    let mut state = State::new();
    let mut out = Vec::with_capacity(ops.len());
    let mut samples: Vec<Vec<RawStyleSample>> = vec![Vec::new(); removal_rects.len()];

    for op in ops {
        match op.operator.as_str() {
            // Graphics state stack.
            "q" => state.push(),
            "Q" => state.pop(),
            "cm" => {
                if let Some(m) = matrix_from_operands(&op.operands) {
                    state.concat_ctm(m);
                }
            }
            // Text object boundaries.
            "BT" => state.begin_text(),
            "ET" => state.end_text(),
            // Text state.
            "Tf" => {
                if let Some(Object::Name(name)) = op.operands.first() {
                    state.current.font_resource = Some(name.clone());
                }
                if let Some(size) = op.operands.get(1).and_then(object_as_f32) {
                    state.font_size = size;
                }
            }
            "Tc" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.char_spacing = v;
                }
            }
            "Tw" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.word_spacing = v;
                }
            }
            "Tz" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.horizontal_scaling = v / 100.0;
                }
            }
            // Non-stroking fill colour. We sample only the most common color
            // spaces; rg/g/k cover ~all real PDFs.
            "rg" => {
                if let (Some(r), Some(g), Some(b)) = (
                    op.operands.first().and_then(object_as_f32),
                    op.operands.get(1).and_then(object_as_f32),
                    op.operands.get(2).and_then(object_as_f32),
                ) {
                    state.current.fill_rgb = (r, g, b);
                }
            }
            "g" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.current.fill_rgb = (v, v, v);
                }
            }
            "k" => {
                if let (Some(c), Some(m), Some(y), Some(k)) = (
                    op.operands.first().and_then(object_as_f32),
                    op.operands.get(1).and_then(object_as_f32),
                    op.operands.get(2).and_then(object_as_f32),
                    op.operands.get(3).and_then(object_as_f32),
                ) {
                    // Naive CMYK→RGB; good enough for catching black text.
                    let r = (1.0 - c) * (1.0 - k);
                    let g = (1.0 - m) * (1.0 - k);
                    let b = (1.0 - y) * (1.0 - k);
                    state.current.fill_rgb = (r, g, b);
                }
            }
            "Tm" => {
                if let Some(m) = matrix_from_operands(&op.operands) {
                    state.set_tm(m);
                }
            }
            "Td" | "TD" => {
                if let (Some(tx), Some(ty)) = (
                    op.operands.first().and_then(object_as_f32),
                    op.operands.get(1).and_then(object_as_f32),
                ) {
                    state.move_text(tx, ty);
                    if op.operator == "TD" {
                        state.text_leading = -ty;
                    }
                }
            }
            "TL" => {
                if let Some(leading) = op.operands.first().and_then(object_as_f32) {
                    state.text_leading = leading;
                }
            }
            "T*" => {
                let leading = state.text_leading;
                state.move_text(0.0, -leading);
            }
            // Text-show operators.
            "Tj" | "TJ" | "'" | "\"" => {
                if op.operator == "'" {
                    let leading = state.text_leading;
                    state.move_text(0.0, -leading);
                } else if op.operator == "\"" {
                    if let Some(word_spacing) = op.operands.first().and_then(object_as_f32) {
                        state.word_spacing = word_spacing;
                    }
                    if let Some(char_spacing) = op.operands.get(1).and_then(object_as_f32) {
                        state.char_spacing = char_spacing;
                    }
                    let leading = state.text_leading;
                    state.move_text(0.0, -leading);
                }

                let origin = state.current_text_origin();
                let dropped_rect = removal_rects
                    .iter()
                    .position(|r| r.contains(origin.0, origin.1));
                if let Some(rect_index) = dropped_rect {
                    let combined = state.text_matrix.mul(state.current.ctm);
                    // Combined `text_matrix × CTM` carries both the producer's
                    // glyph orientation *and* whatever scale the CTM applied
                    // (a `.75 0 0 .75 ... cm` pre-scale is common). Split:
                    // store the orientation as a unit-length rotation/flip,
                    // and bake the scale into `font_size` so the value we
                    // record is the effective user-space size.
                    let x_scale = (combined.a * combined.a + combined.b * combined.b).sqrt();
                    let y_scale = (combined.c * combined.c + combined.d * combined.d).sqrt();
                    let safe_x = if x_scale > 1e-6 { x_scale } else { 1.0 };
                    let safe_y = if y_scale > 1e-6 { y_scale } else { 1.0 };
                    samples[rect_index].push(RawStyleSample {
                        font_resource: state.current.font_resource.clone(),
                        fill_rgb: state.current.fill_rgb,
                        font_size: state.font_size * safe_x,
                        origin,
                        text_orientation: (
                            combined.a / safe_x,
                            combined.b / safe_x,
                            combined.c / safe_y,
                            combined.d / safe_y,
                        ),
                    });
                    let advance = text_show_advance(&op, &state, font_advances);
                    state.advance_text(advance);
                    // Drop the op after advancing the text cursor, so any
                    // later show operators in the same BT/ET block are tested
                    // at their true origin.
                    continue;
                }
                let advance = text_show_advance(&op, &state, font_advances);
                state.advance_text(advance);
            }
            _ => {}
        }
        out.push(op);
    }

    (out, state.current.ctm, samples)
}

// ---------------------------------------------------------------------------
// Graphics-state machine (just enough to compute text origin in user space).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Matrix {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
}

impl Matrix {
    fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    fn translate(tx: f32, ty: f32) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: tx,
            f: ty,
        }
    }

    /// PDF matrix multiplication: `self * other` in column-vector convention,
    /// or equivalently `other` left-multiplied by `self` for row-vectors.
    /// Concretely: result transforms a point as `point -> other -> self`.
    fn mul(self, other: Matrix) -> Matrix {
        Matrix {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            e: self.e * other.a + self.f * other.c + other.e,
            f: self.e * other.b + self.f * other.d + other.f,
        }
    }

    fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    /// Inverse of an affine PDF matrix. Returns `None` if singular.
    fn inverse(&self) -> Option<Matrix> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-9 {
            return None;
        }
        let inv_a = self.d / det;
        let inv_b = -self.b / det;
        let inv_c = -self.c / det;
        let inv_d = self.a / det;
        let inv_e = -(self.e * inv_a + self.f * inv_c);
        let inv_f = -(self.e * inv_b + self.f * inv_d);
        Some(Matrix {
            a: inv_a,
            b: inv_b,
            c: inv_c,
            d: inv_d,
            e: inv_e,
            f: inv_f,
        })
    }
}

#[derive(Debug, Clone)]
struct GraphicsState {
    ctm: Matrix,
    /// Current non-stroking fill color in RGB (`[0, 1]`).
    fill_rgb: (f32, f32, f32),
    /// Current font resource name in the page's /Font dict (e.g. `b"F1"`).
    font_resource: Option<Vec<u8>>,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            ctm: Matrix::identity(),
            fill_rgb: (0.0, 0.0, 0.0),
            font_resource: None,
        }
    }
}

#[derive(Debug)]
struct State {
    stack: Vec<GraphicsState>,
    current: GraphicsState,
    in_text: bool,
    text_matrix: Matrix,
    text_line_matrix: Matrix,
    font_size: f32,
    char_spacing: f32,
    word_spacing: f32,
    horizontal_scaling: f32,
    text_leading: f32,
}

impl State {
    fn new() -> Self {
        Self {
            stack: Vec::new(),
            current: GraphicsState::default(),
            in_text: false,
            text_matrix: Matrix::identity(),
            text_line_matrix: Matrix::identity(),
            font_size: 12.0,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 1.0,
            text_leading: 0.0,
        }
    }

    fn push(&mut self) {
        self.stack.push(self.current.clone());
    }

    fn pop(&mut self) {
        if let Some(prev) = self.stack.pop() {
            self.current = prev;
        }
    }

    fn concat_ctm(&mut self, m: Matrix) {
        // PDF cm spec: new CTM = m × old CTM
        self.current.ctm = m.mul(self.current.ctm);
    }

    fn begin_text(&mut self) {
        self.in_text = true;
        self.text_matrix = Matrix::identity();
        self.text_line_matrix = Matrix::identity();
    }

    fn end_text(&mut self) {
        self.in_text = false;
    }

    fn set_tm(&mut self, m: Matrix) {
        self.text_matrix = m;
        self.text_line_matrix = m;
    }

    fn move_text(&mut self, tx: f32, ty: f32) {
        // Td: Tlm = translate(tx, ty) × Tlm; Tm = Tlm
        let new_lm = Matrix::translate(tx, ty).mul(self.text_line_matrix);
        self.text_line_matrix = new_lm;
        self.text_matrix = new_lm;
    }

    fn advance_text(&mut self, tx: f32) {
        self.text_matrix = Matrix::translate(tx, 0.0).mul(self.text_matrix);
    }

    fn current_text_origin(&self) -> (f32, f32) {
        // text origin = text_matrix × (0, 0) → then map through CTM.
        let (tx, ty) = self.text_matrix.transform_point(0.0, 0.0);
        self.current.ctm.transform_point(tx, ty)
    }
}

fn text_show_advance(op: &Operation, state: &State, font_advances: &FontAdvanceMap) -> f32 {
    let font = font_advances.get(state.current.font_resource.as_ref());
    let scale = state.horizontal_scaling;

    let mut text_advance = 0.0f32;
    let add_string = |bytes: &[u8]| {
        let (width_1000, glyphs, spaces) = font.string_width_1000(bytes);
        width_1000 * state.font_size / 1000.0
            + state.char_spacing * glyphs as f32
            + state.word_spacing * spaces as f32
    };

    match op.operator.as_str() {
        "Tj" | "'" => {
            if let Some(bytes) = op.operands.last().and_then(|o| o.as_str().ok()) {
                text_advance += add_string(bytes);
            }
        }
        "\"" => {
            if let Some(bytes) = op.operands.get(2).and_then(|o| o.as_str().ok()) {
                text_advance += add_string(bytes);
            }
        }
        "TJ" => {
            if let Some(items) = op.operands.first().and_then(|o| o.as_array().ok()) {
                for item in items {
                    match item {
                        Object::String(bytes, _) => text_advance += add_string(bytes),
                        Object::Integer(_) | Object::Real(_) => {
                            if let Some(adjustment) = object_as_f32(item) {
                                text_advance -= adjustment * state.font_size / 1000.0;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    }

    text_advance * scale
}

fn object_as_f32(obj: &Object) -> Option<f32> {
    match obj {
        Object::Integer(i) => Some(*i as f32),
        Object::Real(r) => Some(*r),
        _ => None,
    }
}

fn matrix_from_operands(operands: &[Object]) -> Option<Matrix> {
    if operands.len() < 6 {
        return None;
    }
    Some(Matrix {
        a: object_as_f32(&operands[0])?,
        b: object_as_f32(&operands[1])?,
        c: object_as_f32(&operands[2])?,
        d: object_as_f32(&operands[3])?,
        e: object_as_f32(&operands[4])?,
        f: object_as_f32(&operands[5])?,
    })
}

// ---------------------------------------------------------------------------
// Translated-text content stream.
// ---------------------------------------------------------------------------

/// Resolve a [`FontMetrics`] for each translated block. Blocks whose
/// `(language, style)` map to the same font file share a parsed face — we
/// memoise per-`FontRequest` and pre-resolve advances for the union of texts
/// using that font, so the wrap loop is a hashmap lookup per character.
///

/// Add every [`EmbeddedFont`] used on a page to its `/Resources/Font` dict
/// (deduplicated by resource name).
fn attach_embedded_fonts_to_page(
    doc: &mut Document,
    page_id: ObjectId,
    embeds: &[Option<crate::pdf_font_embed::EmbeddedFont>],
) -> Result<(), PdfWriteError> {
    let unique: std::collections::HashMap<Vec<u8>, ObjectId> = embeds
        .iter()
        .filter_map(|e| e.as_ref().map(|e| (e.resource_name.clone(), e.type0_id)))
        .collect();
    if unique.is_empty() {
        return Ok(());
    }
    let resources_id = ensure_inline_resources(doc, page_id)?;
    let resources = doc
        .get_object_mut(resources_id)
        .and_then(Object::as_dict_mut)?;
    let font_dict = match resources.get_mut(b"Font") {
        Ok(Object::Dictionary(d)) => d,
        _ => {
            resources.set("Font", Object::Dictionary(Dictionary::new()));
            resources
                .get_mut(b"Font")
                .expect("just inserted")
                .as_dict_mut()
                .expect("just inserted as dict")
        }
    };
    for (name, id) in unique {
        font_dict.set(name, Object::Reference(id));
    }
    Ok(())
}

/// Per-block resolved fonts: one `(FontMetrics, EmbeddedFont)` entry per
/// `(bold, italic)` variant the block actually uses (its dominant style
/// plus whatever appears in `style_spans`). Wrapping always uses the
/// dominant variant for width estimation; emit picks per segment.
struct BlockResources {
    by_flags: std::collections::HashMap<
        (bool, bool),
        (
            crate::font_metrics::FontMetrics,
            Option<crate::pdf_font_embed::EmbeddedFont>,
        ),
    >,
    default_flags: (bool, bool),
    monospace: bool,
}

impl BlockResources {
    fn dominant_metrics(&self) -> &crate::font_metrics::FontMetrics {
        &self
            .by_flags
            .get(&self.default_flags)
            .expect("dominant variant is always inserted")
            .0
    }

    fn for_flags(
        &self,
        flags: (bool, bool),
    ) -> (
        &crate::font_metrics::FontMetrics,
        Option<&crate::pdf_font_embed::EmbeddedFont>,
    ) {
        let entry = self
            .by_flags
            .get(&flags)
            .or_else(|| self.by_flags.get(&self.default_flags))
            .expect("at least the dominant variant exists");
        (&entry.0, entry.1.as_ref())
    }
}

fn build_overlay_stream(
    blocks: &[TranslatedStyledBlock],
    user_rects: &[UserRect],
    block_styles: &[SampledBlockStyle],
    block_resources: &[BlockResources],
    geom: PageGeometry,
    final_ctm: Matrix,
) -> Vec<u8> {
    let mut out = Vec::<u8>::new();
    out.extend_from_slice(b"q\n");
    let inv_ctm = final_ctm.inverse().unwrap_or_else(Matrix::identity);
    for (i, (block, user_rect)) in blocks.iter().zip(user_rects.iter()).enumerate() {
        let style = block_styles.get(i).cloned().unwrap_or_default();
        let Some(resources) = block_resources.get(i) else {
            continue;
        };
        emit_block(
            &mut out, block, *user_rect, &style, resources, geom, &inv_ctm,
        );
    }
    out.extend_from_slice(b"Q\n");
    out
}

/// Emit one translated block. Positioning happens in PDF user space (which
/// matches what `UserRect` carries), then we inverse-transform through the
/// page's still-active CTM into the producer's local coordinate system so
/// the appended `cm`-less stream draws at the right visual spot.
fn emit_block(
    out: &mut Vec<u8>,
    block: &TranslatedStyledBlock,
    user_rect: UserRect,
    style: &SampledBlockStyle,
    resources: &BlockResources,
    geom: PageGeometry,
    inv_ctm: &Matrix,
) {
    let text = block.text.trim();
    if text.is_empty() {
        return;
    }
    let user_w = user_rect.x1 - user_rect.x0;
    let user_h = user_rect.y1 - user_rect.y0;
    if user_w <= 0.0 || user_h <= 0.0 {
        return;
    }

    let (vis_w, vis_h) = match geom.rotate {
        90 | 270 => (user_h, user_w),
        _ => (user_w, user_h),
    };
    let line_widths = line_available_widths(style, user_rect, geom, vis_w);

    let dominant_metrics = resources.dominant_metrics();
    let (font_size, lines) = match style.font_size {
        Some(size) if size.is_finite() && size > 0.0 => fit_with_sampled_size(
            text,
            &line_widths,
            vis_h,
            size,
            dominant_metrics,
            style.original_line_count,
        ),
        _ => {
            let initial = (vis_h * (1.0 - TEXT_BASELINE_PAD)).max(4.0);
            wrap_to_fit(text, &line_widths, vis_h, initial, dominant_metrics)
        }
    };
    let leading = font_size * LINE_HEIGHT_FACTOR;

    let (line_dx, line_dy) = match geom.rotate {
        0 => (0.0, -1.0),
        90 => (1.0, 0.0),
        180 => (0.0, 1.0),
        270 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    };

    let (first_baseline_x, first_baseline_y) = match style.anchor {
        Some((ax, ay)) => (ax, ay),
        None => {
            let total_height = leading * lines.len() as f32;
            let top_pad = ((vis_h - total_height).max(0.0)) * 0.5;
            let first_baseline_offset = top_pad + font_size;
            let (top_x, top_y) = match geom.rotate {
                0 => (user_rect.x0, user_rect.y1),
                90 => (user_rect.x0, user_rect.y0),
                180 => (user_rect.x1, user_rect.y0),
                270 => (user_rect.x1, user_rect.y1),
                _ => (user_rect.x0, user_rect.y1),
            };
            (
                top_x + first_baseline_offset * line_dx,
                top_y + first_baseline_offset * line_dy,
            )
        }
    };

    let _ = writeln!(
        out,
        "{:.3} {:.3} {:.3} rg",
        style.fill_rgb.0, style.fill_rgb.1, style.fill_rgb.2
    );
    out.extend_from_slice(b"BT\n");

    // Map each wrapped line back to its byte ranges in `block.text` so we
    // can intersect with `block.style_spans` and produce styled segments.
    let line_word_ranges = line_byte_ranges(&block.text, &lines);

    for (i, _line) in lines.iter().enumerate() {
        let (line_x, line_y) = line_origin(
            style,
            i,
            first_baseline_x,
            first_baseline_y,
            leading,
            line_dx,
            line_dy,
        );

        // Per-segment "advance right" vector in user space: row 1 of the
        // sampled orientation matrix. For pure translation that's (1, 0); for
        // 90° rotated producers it's (0, 1); etc.
        let advance_dx = style.text_orientation.0;
        let advance_dy = style.text_orientation.1;

        let segments = segments_for_line(
            &block.text,
            &line_word_ranges[i],
            &block.style_spans,
            resources.default_flags,
        );

        let mut cumulative = 0.0_f32;
        for seg in segments {
            if seg.text.is_empty() {
                continue;
            }
            let (seg_metrics, seg_embed) = resources.for_flags(seg.flags);
            let seg_resource_name: &[u8] = match seg_embed {
                Some(e) => &e.resource_name,
                None => SampledBlockStyle {
                    bold: seg.flags.0,
                    italic: seg.flags.1,
                    monospace: resources.monospace,
                    ..SampledBlockStyle::default()
                }
                .font_resource(),
            };
            let seg_x = line_x + cumulative * advance_dx;
            let seg_y = line_y + cumulative * advance_dy;
            let combined = Matrix {
                a: style.text_orientation.0,
                b: style.text_orientation.1,
                c: style.text_orientation.2,
                d: style.text_orientation.3,
                e: seg_x,
                f: seg_y,
            };
            let tm = combined.mul(*inv_ctm);
            let _ = writeln!(
                out,
                "/{} {:.2} Tf",
                std::str::from_utf8(seg_resource_name).unwrap(),
                font_size
            );
            let _ = writeln!(
                out,
                "{:.4} {:.4} {:.4} {:.4} {:.2} {:.2} Tm",
                tm.a, tm.b, tm.c, tm.d, tm.e, tm.f
            );
            emit_tj_for_segment(out, &seg.text, seg_metrics, seg_embed);

            cumulative += seg_metrics.measure(&seg.text, font_size);
        }
    }
    out.extend_from_slice(b"ET\n\n");
}

fn line_origin(
    style: &SampledBlockStyle,
    line_index: usize,
    first_x: f32,
    first_y: f32,
    leading: f32,
    line_dx: f32,
    line_dy: f32,
) -> (f32, f32) {
    if let Some(anchor) = style.line_anchors.get(line_index) {
        return *anchor;
    }
    if let Some(last) = style.line_anchors.last() {
        let extra = (line_index + 1 - style.line_anchors.len()) as f32 * leading;
        return (last.0 + extra * line_dx, last.1 + extra * line_dy);
    }
    let off = line_index as f32 * leading;
    (first_x + off * line_dx, first_y + off * line_dy)
}

fn line_available_widths(
    style: &SampledBlockStyle,
    user_rect: UserRect,
    geom: PageGeometry,
    fallback: f32,
) -> Vec<f32> {
    if style.line_anchors.is_empty() {
        return vec![fallback.max(1.0)];
    }

    let visual_right = user_rect_visual_bounds(user_rect, geom).2;
    style
        .line_anchors
        .iter()
        .map(|origin| {
            let (vx, _) = visual_position(*origin, geom);
            (visual_right - vx).max(fallback * 0.25).min(fallback)
        })
        .collect()
}

fn user_rect_visual_bounds(rect: UserRect, geom: PageGeometry) -> (f32, f32, f32, f32) {
    let points = [
        (rect.x0, rect.y0),
        (rect.x0, rect.y1),
        (rect.x1, rect.y0),
        (rect.x1, rect.y1),
    ];
    let mut left = f32::INFINITY;
    let mut top = f32::INFINITY;
    let mut right = f32::NEG_INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for point in points {
        let (x, y) = visual_position(point, geom);
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }
    (left, top, right, bottom)
}

/// One run of consecutive characters from a wrapped line that share the
/// same `(bold, italic)` style flags.
struct LineSegment {
    text: String,
    flags: (bool, bool),
}

/// Walk the line's words (located in `block_text` via `word_ranges`),
/// snap each word's style to its **majority** char flag (Bergamot's
/// token alignment often spans whitespace, so going char-by-char produces
/// off-by-one bold edges; snapping to whole words makes bold/italic
/// word-aligned), then group consecutive same-flag words into segments.
fn segments_for_line(
    block_text: &str,
    word_ranges: &[(usize, usize)],
    style_spans: &[crate::styled::StyleSpan],
    default_flags: (bool, bool),
) -> Vec<LineSegment> {
    let lookup = |byte: usize| -> (bool, bool) {
        for span in style_spans {
            if byte >= span.start as usize && byte < span.end as usize {
                if let Some(s) = &span.style {
                    return (s.bold, s.italic);
                }
            }
        }
        default_flags
    };

    // Per-word majority flag.
    let mut word_flags: Vec<(bool, bool)> = Vec::with_capacity(word_ranges.len());
    for (start, end) in word_ranges {
        let mut counts: std::collections::HashMap<(bool, bool), usize> =
            std::collections::HashMap::new();
        let mut byte = *start;
        for c in block_text[*start..*end].chars() {
            *counts.entry(lookup(byte)).or_default() += 1;
            byte += c.len_utf8();
        }
        let majority = counts
            .into_iter()
            .max_by_key(|(_, n)| *n)
            .map(|(f, _)| f)
            .unwrap_or(default_flags);
        word_flags.push(majority);
    }

    // Group consecutive same-flag words into segments. Word-separator
    // spaces stay attached to the *previous* segment so that breaking
    // segments (bold→regular transition) doesn't drop them.
    let mut segments: Vec<LineSegment> = Vec::new();
    let mut current = String::new();
    let mut current_flags = default_flags;
    for (i, ((start, end), flags)) in word_ranges.iter().zip(word_flags.iter()).enumerate() {
        let word = &block_text[*start..*end];
        // Bergamot's SentencePiece detokenizer emits a space before
        // closing punctuation (`,`, `.`, `)`, etc.) — visually fine when
        // the surrounding text shares one font, but the bold transitions
        // we now add make the gap obvious. Suppress the separator before
        // any token that starts with a closing-punctuation glyph.
        let hugs_previous = word
            .chars()
            .next()
            .is_some_and(|c| matches!(c, ',' | '.' | ')' | ']' | '}' | ':' | ';' | '?' | '!'));
        let separator = i > 0 && !hugs_previous;
        let need_break = !current.is_empty() && *flags != current_flags;
        if need_break {
            if separator {
                current.push(' ');
            }
            segments.push(LineSegment {
                text: std::mem::take(&mut current),
                flags: current_flags,
            });
            current_flags = *flags;
        } else if separator {
            current.push(' ');
        }
        if current.is_empty() {
            current_flags = *flags;
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        segments.push(LineSegment {
            text: current,
            flags: current_flags,
        });
    }
    segments
}

/// Locate each wrapped line's word byte ranges back inside `block_text`.
/// `wrap_lines` produces `Vec<String>` whose words appear in order in the
/// source; we forward-scan, skipping whitespace, matching each word.
fn line_byte_ranges(block_text: &str, lines: &[String]) -> Vec<Vec<(usize, usize)>> {
    let mut cursor = 0usize;
    let mut all = Vec::with_capacity(lines.len());
    for line in lines {
        let mut line_ranges = Vec::new();
        for word in line.split_whitespace() {
            // Skip whitespace.
            while cursor < block_text.len() {
                let c = match block_text[cursor..].chars().next() {
                    Some(c) => c,
                    None => break,
                };
                if c.is_whitespace() {
                    cursor += c.len_utf8();
                } else {
                    break;
                }
            }
            let word_bytes = word.as_bytes();
            let end = cursor + word_bytes.len();
            if end <= block_text.len() && &block_text.as_bytes()[cursor..end] == word_bytes {
                line_ranges.push((cursor, end));
                cursor = end;
            }
            // If mismatch (shouldn't happen since wrap_lines preserves words),
            // we just skip and keep going. Style attribution may be slightly
            // off for that one word.
        }
        all.push(line_ranges);
    }
    all
}

fn emit_tj_for_segment(
    out: &mut Vec<u8>,
    text: &str,
    metrics: &crate::font_metrics::FontMetrics,
    embed: Option<&crate::pdf_font_embed::EmbeddedFont>,
) {
    if let Some(embedded) = embed {
        out.push(b'<');
        for c in text.chars() {
            let original = metrics.glyph_for(c).map(|g| g.gid).unwrap_or(0);
            let gid = embedded
                .gid_remap
                .get(&original)
                .copied()
                .unwrap_or(original);
            let _ = write!(out, "{:04X}", gid);
        }
        out.extend_from_slice(b"> Tj\n");
    } else {
        out.extend_from_slice(b"(");
        write_pdf_string_body(out, text);
        out.extend_from_slice(b") Tj\n");
    }
}

fn write_pdf_string_body(out: &mut Vec<u8>, text: &str) {
    for c in text.chars() {
        let byte = unicode_to_winansi(c);
        match byte {
            b'(' | b')' | b'\\' => {
                out.push(b'\\');
                out.push(byte);
            }
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            _ => out.push(byte),
        }
    }
}

/// Tolerated horizontal overhang past the bbox right edge before we shrink
/// the font (covers the case where our 0.5em advance is mildly pessimistic
/// against the real Helvetica metrics).
const OVERHANG_TOLERANCE: f32 = 1.05;
/// Floor for shrink-to-fit (fraction of the sampled size). Below this the
/// text becomes unreadable so we accept overflow instead.
const MIN_SHRINK_FRACTION: f32 = 0.7;

/// Fit `text` inside (`vis_w`, `vis_h`) starting from the original sampled
/// `font_size`. Distinguishes single-line from multi-line originals and
/// chooses between width-shrink (single-line) and wrap-then-shrink
/// (multi-line). Refuses to shrink below `MIN_SHRINK_FRACTION` of the
/// sampled size — beyond that we'd produce unreadable output.
fn fit_with_sampled_size(
    text: &str,
    line_widths: &[f32],
    vis_h: f32,
    sampled: f32,
    metrics: &crate::font_metrics::FontMetrics,
    original_lines: usize,
) -> (f32, Vec<String>) {
    let min_size = (sampled * MIN_SHRINK_FRACTION).max(4.0);

    if original_lines <= 1 {
        // Originally one line. Don't introduce wraps — keep one line and
        // shrink the font if it'd overflow the bbox by more than tolerance.
        let vis_w = line_width_at(line_widths, 0);
        let width_at_sampled = metrics.measure(text, sampled);
        let allowed = vis_w * OVERHANG_TOLERANCE;
        let final_size = if width_at_sampled <= allowed || width_at_sampled == 0.0 {
            sampled
        } else {
            (sampled * vis_w / width_at_sampled).max(min_size)
        };
        (final_size, vec![text.to_string()])
    } else {
        // Originally multi-line. Wrap at the sampled size, and if the
        // wrap produces more lines than the original used, shrink and
        // re-wrap. Targeting the original line count is better than
        // checking against `vis_h` because the producer's column width is
        // what actually mattered: bbox height is just the union of glyph
        // ink and may include slack at top/bottom.
        let mut size = sampled;
        let mut lines = wrap_lines_to_widths(text, line_widths, size, metrics);
        for _ in 0..6 {
            if lines.len() <= original_lines || size <= min_size {
                break;
            }
            size = (size * 0.9).max(min_size);
            lines = wrap_lines_to_widths(text, line_widths, size, metrics);
        }
        // Final safety: if even at min_size we'd still exceed the bbox
        // height substantially, accept it — at least the text is readable.
        let _ = vis_h;
        (size, lines)
    }
}

fn wrap_to_fit(
    text: &str,
    line_widths: &[f32],
    max_height: f32,
    mut font_size: f32,
    metrics: &crate::font_metrics::FontMetrics,
) -> (f32, Vec<String>) {
    for _ in 0..6 {
        let lines = wrap_lines_to_widths(text, line_widths, font_size, metrics);
        let total_height = font_size * LINE_HEIGHT_FACTOR * lines.len() as f32;
        if total_height <= max_height || font_size <= 4.0 {
            return (font_size, lines);
        }
        font_size *= 0.85;
    }
    let final_size = font_size.max(4.0);
    (
        final_size,
        wrap_lines_to_widths(text, line_widths, final_size, metrics),
    )
}

#[cfg(test)]
fn wrap_lines(
    text: &str,
    max_width: f32,
    font_size: f32,
    metrics: &crate::font_metrics::FontMetrics,
) -> Vec<String> {
    wrap_lines_to_widths(text, &[max_width], font_size, metrics)
}

fn wrap_lines_to_widths(
    text: &str,
    line_widths: &[f32],
    font_size: f32,
    metrics: &crate::font_metrics::FontMetrics,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        let max_width = line_width_at(line_widths, lines.len());
        if metrics.measure(&candidate, font_size) <= max_width || current.is_empty() {
            current = candidate;
        } else {
            lines.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn line_width_at(line_widths: &[f32], index: usize) -> f32 {
    line_widths
        .get(index)
        .copied()
        .or_else(|| line_widths.last().copied())
        .unwrap_or(1.0)
        .max(1.0)
}

/// Map a Unicode codepoint to its WinAnsi (CP1252) byte, or `b'?'` if there
/// is no WinAnsi codepoint for it.
fn unicode_to_winansi(c: char) -> u8 {
    match c as u32 {
        0x00..=0x7F => c as u8,
        0x20AC => 0x80, // €
        0x201A => 0x82,
        0x0192 => 0x83,
        0x201E => 0x84,
        0x2026 => 0x85,
        0x2020 => 0x86,
        0x2021 => 0x87,
        0x02C6 => 0x88,
        0x2030 => 0x89,
        0x0160 => 0x8A,
        0x2039 => 0x8B,
        0x0152 => 0x8C,
        0x017D => 0x8E,
        0x2018 => 0x91,
        0x2019 => 0x92,
        0x201C => 0x93,
        0x201D => 0x94,
        0x2022 => 0x95,
        0x2013 => 0x96,
        0x2014 => 0x97,
        0x02DC => 0x98,
        0x2122 => 0x99,
        0x0161 => 0x9A,
        0x203A => 0x9B,
        0x0153 => 0x9C,
        0x017E => 0x9E,
        0x0178 => 0x9F,
        0xA0..=0xFF => c as u8,
        _ => b'?',
    }
}

// ---------------------------------------------------------------------------
// Resource and content-array housekeeping.
// ---------------------------------------------------------------------------

fn ensure_fonts_in_page_resources(
    doc: &mut Document,
    page_id: ObjectId,
) -> Result<(), PdfWriteError> {
    // Register Helvetica + Courier variants so we can mimic bold / italic /
    // monospace styles sampled from the original. All eight are PDF
    // standard-14 base fonts; no embedding needed.
    let variants: [(&[u8], &[u8]); 8] = [
        (HELVETICA_REGULAR, b"Helvetica"),
        (HELVETICA_BOLD, b"Helvetica-Bold"),
        (HELVETICA_OBLIQUE, b"Helvetica-Oblique"),
        (HELVETICA_BOLD_OBLIQUE, b"Helvetica-BoldOblique"),
        (COURIER_REGULAR, b"Courier"),
        (COURIER_BOLD, b"Courier-Bold"),
        (COURIER_OBLIQUE, b"Courier-Oblique"),
        (COURIER_BOLD_OBLIQUE, b"Courier-BoldOblique"),
    ];
    let mut new_refs = Vec::with_capacity(variants.len());
    for (resource_name, base_font) in variants {
        let id = doc.add_object({
            let mut d = Dictionary::new();
            d.set("Type", Object::Name(b"Font".to_vec()));
            d.set("Subtype", Object::Name(b"Type1".to_vec()));
            d.set("BaseFont", Object::Name(base_font.to_vec()));
            d.set("Encoding", Object::Name(b"WinAnsiEncoding".to_vec()));
            Object::Dictionary(d)
        });
        new_refs.push((resource_name, id));
    }

    let resources_id = ensure_inline_resources(doc, page_id)?;
    let resources = doc
        .get_object_mut(resources_id)
        .and_then(Object::as_dict_mut)?;

    let font_dict = match resources.get_mut(b"Font") {
        Ok(Object::Dictionary(d)) => d,
        _ => {
            resources.set("Font", Object::Dictionary(Dictionary::new()));
            resources
                .get_mut(b"Font")
                .expect("just inserted")
                .as_dict_mut()
                .expect("just inserted as dict")
        }
    };
    for (resource_name, id) in new_refs {
        font_dict.set(resource_name, Object::Reference(id));
    }
    Ok(())
}

fn ensure_inline_resources(
    doc: &mut Document,
    page_id: ObjectId,
) -> Result<ObjectId, PdfWriteError> {
    if let Ok(page) = doc.get_object(page_id).and_then(Object::as_dict) {
        if let Ok(Object::Reference(id)) = page.get(b"Resources") {
            return Ok(*id);
        }
    }

    let inline_resources = effective_page_resources(doc, page_id)?;

    let new_id = doc.add_object(Object::Dictionary(inline_resources));
    let page_mut = doc.get_object_mut(page_id).and_then(Object::as_dict_mut)?;
    page_mut.set("Resources", Object::Reference(new_id));
    Ok(new_id)
}

fn effective_page_resources(
    doc: &Document,
    page_id: ObjectId,
) -> Result<Dictionary, PdfWriteError> {
    let mut current_id = page_id;
    let mut seen = std::collections::HashSet::new();

    loop {
        if !seen.insert(current_id) {
            return Err(PdfWriteError::Other(format!(
                "cycle while resolving page resources at object {current_id:?}"
            )));
        }

        let node = doc.get_object(current_id).and_then(Object::as_dict)?;
        if let Ok(resources) = node.get(b"Resources") {
            return match resources {
                Object::Dictionary(d) => Ok(d.clone()),
                Object::Reference(id) => Ok(doc.get_dictionary(*id)?.clone()),
                _ => Ok(Dictionary::new()),
            };
        }

        let Ok(parent_id) = node.get(b"Parent").and_then(Object::as_reference) else {
            return Ok(Dictionary::new());
        };
        current_id = parent_id;
    }
}

/// Strip every `/Resources/Font` entry that no surviving `Tj`/`TJ`/`'`/`"`
/// references, then garbage-collect orphaned font dicts and their embedded
/// font streams. This is what reclaims the original PDF's font payload —
/// surgery removed the *operators* that drew the original glyphs but left
/// the font dictionaries reachable through the resources dict.
fn prune_unused_fonts(
    doc: &mut Document,
    modified_pages: &std::collections::HashSet<ObjectId>,
) -> Result<(), PdfWriteError> {
    for page_id in modified_pages.iter().copied() {
        let used = used_font_resource_names(doc, page_id)?;
        let resources_id = ensure_inline_resources(doc, page_id)?;
        let resources = doc
            .get_object_mut(resources_id)
            .and_then(Object::as_dict_mut)?;
        let to_remove: Vec<Vec<u8>> = match resources.get(b"Font") {
            Ok(Object::Dictionary(d)) => d
                .iter()
                .filter_map(|(k, _)| {
                    (is_translator_font_resource(k) && !used.contains(k)).then(|| k.clone())
                })
                .collect(),
            _ => Vec::new(),
        };
        if to_remove.is_empty() {
            continue;
        }
        if let Ok(Object::Dictionary(font_dict)) = resources.get_mut(b"Font") {
            for k in &to_remove {
                font_dict.remove(k);
            }
        }
    }
    // Now that no /Resources/Font entry references them, the font dicts and
    // their /FontFile streams are unreachable from /Root and prune_objects()
    // collects them.
    doc.prune_objects();
    Ok(())
}

fn is_translator_font_resource(name: &[u8]) -> bool {
    name.starts_with(b"Tr")
}

fn used_font_resource_names(
    doc: &Document,
    page_id: ObjectId,
) -> Result<std::collections::HashSet<Vec<u8>>, PdfWriteError> {
    let content = doc.get_and_decode_page_content(page_id)?;
    let mut used = std::collections::HashSet::new();
    let mut current: Option<Vec<u8>> = None;
    for op in &content.operations {
        match op.operator.as_str() {
            "Tf" => {
                if let Some(Object::Name(n)) = op.operands.first() {
                    current = Some(n.clone());
                }
            }
            // Any text-show operator counts the active font as used.
            "Tj" | "TJ" | "'" | "\"" => {
                if let Some(name) = &current {
                    used.insert(name.clone());
                }
            }
            _ => {}
        }
    }
    Ok(used)
}

fn append_content_stream(
    doc: &mut Document,
    page_id: ObjectId,
    stream_bytes: Vec<u8>,
) -> Result<(), PdfWriteError> {
    let new_stream_id =
        doc.add_object(Object::Stream(Stream::new(Dictionary::new(), stream_bytes)));

    let page_mut = doc.get_object_mut(page_id).and_then(Object::as_dict_mut)?;
    let new_contents = match page_mut.get(b"Contents") {
        Ok(Object::Reference(existing_id)) => Object::Array(vec![
            Object::Reference(*existing_id),
            Object::Reference(new_stream_id),
        ]),
        Ok(Object::Array(existing)) => {
            let mut arr = existing.clone();
            arr.push(Object::Reference(new_stream_id));
            Object::Array(arr)
        }
        _ => Object::Reference(new_stream_id),
    };
    page_mut.set("Contents", new_contents);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_helper(text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        write_pdf_string_body(&mut out, text);
        out
    }

    #[test]
    fn encodes_basic_latin() {
        assert_eq!(encode_helper("Hola"), b"Hola");
        assert_eq!(
            encode_helper("á é í ó ú ñ"),
            b"\xE1 \xE9 \xED \xF3 \xFA \xF1"
        );
        assert_eq!(encode_helper("(parens)"), b"\\(parens\\)");
        assert_eq!(encode_helper("back\\slash"), b"back\\\\slash");
    }

    #[test]
    fn encodes_euro_as_single_byte() {
        assert_eq!(encode_helper("€100"), b"\x80100");
    }

    #[test]
    fn replaces_unmappable_codepoints() {
        assert_eq!(encode_helper("日本"), b"??");
    }

    #[test]
    fn wraps_long_text_into_multiple_lines() {
        let text = "the quick brown fox jumps over the lazy dog repeatedly";
        let metrics = crate::font_metrics::FontMetrics::approx(HELVETICA_AVG_ADVANCE);
        let lines = wrap_lines(text, 60.0, 10.0, &metrics);
        assert!(lines.len() > 1);
        for line in &lines {
            if line.contains(' ') {
                let w = metrics.measure(line, 10.0);
                assert!(w <= 60.0, "line too wide: {line:?} width {w}");
            }
        }
    }

    #[test]
    fn empty_translations_does_not_touch_pdf() {
        let result = write_translated_pdf(b"", &[], &crate::font_provider::NoFontProvider);
        assert!(result.is_err());
    }

    #[test]
    fn helvetica_font_dict_shape() {
        let mut doc = Document::with_version("1.5");
        let page_id = doc.add_object({
            let mut d = Dictionary::new();
            d.set("Type", Object::Name(b"Page".to_vec()));
            Object::Dictionary(d)
        });
        ensure_fonts_in_page_resources(&mut doc, page_id).unwrap();

        let page = doc.get_object(page_id).unwrap().as_dict().unwrap();
        let resources_ref = page.get(b"Resources").unwrap();
        let resources_id = resources_ref.as_reference().unwrap();
        let resources = doc.get_object(resources_id).unwrap().as_dict().unwrap();
        let fonts = resources.get(b"Font").unwrap().as_dict().unwrap();
        let helv_ref = fonts.get(HELVETICA_REGULAR).unwrap();
        let helv_id = helv_ref.as_reference().unwrap();
        let helv = doc.get_object(helv_id).unwrap().as_dict().unwrap();
        assert_eq!(
            helv.get(b"BaseFont").unwrap().as_name().unwrap(),
            b"Helvetica"
        );
    }

    #[test]
    fn matrix_mul_identity() {
        let i = Matrix::identity();
        let t = Matrix::translate(10.0, 20.0);
        let r = t.mul(i);
        assert!((r.e - 10.0).abs() < 1e-5);
        assert!((r.f - 20.0).abs() < 1e-5);
    }

    #[test]
    fn filter_drops_text_show_inside_rect() {
        // Tiny program: BT, Tf, Td to (100, 700), Tj "hi", ET
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(12.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(b"hi".to_vec(), lopdf::StringFormat::Literal)],
            ),
            Operation::new("ET", vec![]),
        ];

        let rect = UserRect {
            x0: 50.0,
            y0: 650.0,
            x1: 200.0,
            y1: 750.0,
        };
        let (filtered, _, _) = filter_text_ops(ops.clone(), &[rect], &FontAdvanceMap::default());
        // BT, Tf, Td, ET — the Tj should have been dropped.
        assert_eq!(filtered.len(), 4);
        assert!(filtered.iter().all(|o| o.operator != "Tj"));
    }

    #[test]
    fn filter_keeps_text_show_outside_rect() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(12.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(b"hi".to_vec(), lopdf::StringFormat::Literal)],
            ),
            Operation::new("ET", vec![]),
        ];

        let rect = UserRect {
            x0: 0.0,
            y0: 0.0,
            x1: 50.0,
            y1: 50.0,
        };
        let (filtered, _, _) = filter_text_ops(ops, &[rect], &FontAdvanceMap::default());
        assert!(filtered.iter().any(|o| o.operator == "Tj"));
    }

    #[test]
    fn filter_advances_between_consecutive_text_shows() {
        let ops = vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), Object::Real(10.0)]),
            Operation::new("Td", vec![Object::Real(100.0), Object::Real(700.0)]),
            Operation::new(
                "Tj",
                vec![Object::String(
                    b"hello".to_vec(),
                    lopdf::StringFormat::Literal,
                )],
            ),
            Operation::new(
                "Tj",
                vec![Object::String(b"!".to_vec(), lopdf::StringFormat::Literal)],
            ),
            Operation::new("ET", vec![]),
        ];
        let rect = UserRect {
            x0: 124.0,
            y0: 690.0,
            x1: 130.0,
            y1: 710.0,
        };
        let (filtered, _, _) = filter_text_ops(ops, &[rect], &FontAdvanceMap::default());
        assert_eq!(filtered.iter().filter(|op| op.operator == "Tj").count(), 1);
    }

    #[test]
    fn ensure_inline_resources_clones_inherited_resources() {
        let mut doc = Document::with_version("1.5");
        let resources_id = doc.add_object({
            let mut d = Dictionary::new();
            d.set("XObject", Object::Dictionary(Dictionary::new()));
            Object::Dictionary(d)
        });
        let pages_id = doc.add_object({
            let mut d = Dictionary::new();
            d.set("Type", Object::Name(b"Pages".to_vec()));
            d.set("Resources", Object::Reference(resources_id));
            d.set("Kids", Object::Array(Vec::new()));
            d.set("Count", Object::Integer(1));
            Object::Dictionary(d)
        });
        let page_id = doc.add_object({
            let mut d = Dictionary::new();
            d.set("Type", Object::Name(b"Page".to_vec()));
            d.set("Parent", Object::Reference(pages_id));
            Object::Dictionary(d)
        });

        let local_resources_id = ensure_inline_resources(&mut doc, page_id).unwrap();
        assert_ne!(local_resources_id, resources_id);
        let local = doc
            .get_object(local_resources_id)
            .unwrap()
            .as_dict()
            .unwrap();
        assert!(matches!(local.get(b"XObject"), Ok(Object::Dictionary(_))));
    }

    #[test]
    fn detects_style_from_basefont_name() {
        assert_eq!(detect_from_name(b"ArialMT"), (false, false, false));
        assert_eq!(detect_from_name(b"Arial-BoldMT"), (true, false, false));
        assert_eq!(detect_from_name(b"Arial-ItalicMT"), (false, true, false));
        assert_eq!(detect_from_name(b"Arial-BoldItalicMT"), (true, true, false));
        assert_eq!(detect_from_name(b"Helvetica"), (false, false, false));
        assert_eq!(detect_from_name(b"Helvetica-Bold"), (true, false, false));
        assert_eq!(detect_from_name(b"Helvetica-Oblique"), (false, true, false));
        assert_eq!(
            detect_from_name(b"Helvetica-BoldOblique"),
            (true, true, false)
        );
        // Subset-tag prefix handling.
        assert_eq!(
            detect_from_name(b"AAAAAA+Helvetica-Bold"),
            (true, false, false)
        );
        // Monospace detection.
        assert_eq!(detect_from_name(b"Courier"), (false, false, true));
        assert_eq!(detect_from_name(b"Courier-Bold"), (true, false, true));
        assert_eq!(detect_from_name(b"Courier-BoldOblique"), (true, true, true));
        assert_eq!(detect_from_name(b"Consolas"), (false, false, true));
        assert_eq!(
            detect_from_name(b"JetBrainsMono-Regular"),
            (false, false, true)
        );
        assert_eq!(
            detect_from_name(b"SourceCodePro-Regular"),
            (false, false, true)
        );
    }

    #[test]
    fn matrix_inverse_of_rotation() {
        // The producer rotation found in test PDF.
        let m = Matrix {
            a: 0.0,
            b: 1.0,
            c: -1.0,
            d: 0.0,
            e: 595.0,
            f: 0.0,
        };
        let inv = m.inverse().unwrap();
        // Apply m then inv to a sample point — should round-trip.
        let (ux, uy) = m.transform_point(20.0, 484.83);
        let (lx, ly) = inv.transform_point(ux, uy);
        assert!((lx - 20.0).abs() < 1e-3);
        assert!((ly - 484.83).abs() < 1e-3);
    }
}

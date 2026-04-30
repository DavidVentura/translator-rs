//! Translated-text overlay emission.
//!
//! Take the translated blocks plus the per-block typography & geometry
//! sampled by surgery, wrap text to fit, and emit a content stream that
//! draws the new glyphs at the original baselines through a
//! [`ContentStreamBuilder`].

use std::collections::HashMap;

use crate::font_metrics::FontMetrics;
use crate::pdf_content::{
    BoldItalic, ContentStreamBuilder, FontStyleFlags, Matrix, PageGeometry, UserRect,
};
use crate::pdf_font_embed::EmbeddedFont;
use crate::pdf_surgery::CapturedTextShow;
use crate::pdf_write::{BlockGeometry, BlockTypography, SampledBlockStyle};
use crate::styled::{StyleSpan, TranslatedStyledBlock};

/// Approximate average Helvetica glyph width as a fraction of font size.
pub(crate) const HELVETICA_AVG_ADVANCE: f32 = 0.5;

/// Approximate average Courier glyph width as a fraction of font size. Courier
/// is monospaced at a known em-fraction (~0.6em), so this is tighter than the
/// Helvetica figure but still a fallback for when no real font is available.
pub(crate) const COURIER_AVG_ADVANCE: f32 = 0.6;

/// Vertical margin inside the bbox so descenders don't clip the bottom.
const TEXT_BASELINE_PAD: f32 = 0.2;

/// Leading multiplier for wrapped lines (line-height = font_size * factor).
const LINE_HEIGHT_FACTOR: f32 = 1.15;

/// Iterations of shrink-and-rewrap when fitting translated text. Six steps at
/// the chosen factors take any sampled size down to ~MIN_SHRINK_FRACTION
/// before bottoming out.
const FIT_RETRY_LIMIT: usize = 6;

/// Per-iteration shrink factor when the wrap produced more lines than the
/// original.
const MULTILINE_SHRINK_FACTOR: f32 = 0.9;

/// Per-iteration shrink factor when the unwrapped block exceeds vis_h.
const UNWRAPPED_SHRINK_FACTOR: f32 = 0.85;

/// Floor font size during shrink-to-fit. Below this the text becomes
/// unreadable so we accept overflow instead.
const MIN_FIT_FONT_SIZE_PT: f32 = 4.0;

/// Tolerated horizontal overhang past the bbox right edge before we shrink
/// the font (covers the case where our 0.5em advance is mildly pessimistic
/// against the real Helvetica metrics).
const OVERHANG_TOLERANCE: f32 = 1.05;
/// Floor for shrink-to-fit (fraction of the sampled size). Below this the
/// text becomes unreadable so we accept overflow instead.
const MIN_SHRINK_FRACTION: f32 = 0.7;

/// Gap to reserve before an inline opaque formula when wrapping the prose
/// line that precedes it.
const INLINE_FORMULA_PAD_PT: f32 = 6.0;

/// Per-block resolved fonts: one `(FontMetrics, EmbeddedFont)` entry per
/// [`BoldItalic`] variant the block actually uses (its dominant style plus
/// whatever appears in `style_spans`). Wrapping always uses the dominant
/// variant for width estimation; emit picks per segment.
pub(crate) struct BlockResources {
    pub(crate) by_flags: HashMap<BoldItalic, (FontMetrics, Option<EmbeddedFont>)>,
    pub(crate) default_flags: BoldItalic,
    pub(crate) monospace: bool,
}

impl BlockResources {
    pub(crate) fn dominant_metrics(&self) -> &FontMetrics {
        &self
            .by_flags
            .get(&self.default_flags)
            .expect("dominant variant is always inserted")
            .0
    }

    pub(crate) fn for_flags(&self, flags: BoldItalic) -> (&FontMetrics, Option<&EmbeddedFont>) {
        let entry = self
            .by_flags
            .get(&flags)
            .or_else(|| self.by_flags.get(&self.default_flags))
            .expect("at least the dominant variant exists");
        (&entry.0, entry.1.as_ref())
    }
}

pub(crate) fn build_overlay_stream(
    blocks: &[TranslatedStyledBlock],
    user_rects: &[UserRect],
    block_styles: &[SampledBlockStyle],
    block_resources: &[BlockResources],
    captured_text: &[Vec<CapturedTextShow>],
    geom: PageGeometry,
    final_ctm: Matrix,
) -> Vec<u8> {
    let mut builder = ContentStreamBuilder::new();
    builder.save_state();
    let inv_ctm = final_ctm.inverse().unwrap_or_else(Matrix::identity);
    // Cursor coordination across adjacent blocks. After each block we
    // record where its last visual line ended in user space; the next
    // block, if its first visual line is on the same row, picks up from
    // there. That's how a `<text><formula><text>` row stays coherent
    // after translation: the formula slides over to follow the actual
    // prose endpoint, the following prose slides further, and a
    // multi-line prose's later wraps fall back to their own anchors at
    // the left margin.
    let mut prev_line_end: Option<(f32, f32)> = None;
    let mut i = 0usize;
    while i < blocks.len() && i < user_rects.len() {
        let block = &blocks[i];
        let user_rect = user_rects[i];
        let style = block_styles.get(i).cloned().unwrap_or_default();
        let first_line_y = first_line_user_y(&style, user_rect);
        let inherit_x = prev_line_end.and_then(|(px, py)| {
            ((first_line_y - py).abs() <= 4.0).then_some(px + INLINE_FORMULA_PAD_PT)
        });

        if block.opaque {
            let captured = captured_text.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let _ = inherit_x; // formulas stay anchored — see emit_captured_text_block
            prev_line_end =
                emit_captured_text_block(&mut builder, captured, user_rect, &style, &inv_ctm);
            i += 1;
            continue;
        }

        let Some(resources) = block_resources.get(i) else {
            prev_line_end = None;
            i += 1;
            continue;
        };
        // The prose's last visual line might be on the same row as a
        // following opaque formula. Find the leftmost opaque block on
        // that row so the wrap doesn't run past it (formulas stay at
        // original x, so the prose has to leave them room).
        let last_line_obstacle =
            following_inline_obstacle_x(i, blocks, user_rects, &block_styles, geom);
        prev_line_end = emit_block(
            &mut builder,
            block,
            user_rect,
            &style,
            resources,
            inherit_x,
            last_line_obstacle,
            geom,
            &inv_ctm,
        );
        i += 1;
    }
    builder.restore_state();
    builder.finish()
}

fn following_inline_obstacles(
    index: usize,
    blocks: &[TranslatedStyledBlock],
    user_rects: &[UserRect],
    geom: PageGeometry,
) -> Vec<UserRect> {
    let Some(current) = user_rects.get(index).copied() else {
        return Vec::new();
    };
    let (_, current_top, _, current_bottom) = user_rect_visual_bounds(current, geom);
    if current_bottom - current_top > 45.0 {
        return Vec::new();
    }
    let mut obstacles = Vec::new();
    for next in (index + 1)..blocks.len().min(user_rects.len()) {
        let rect = user_rects[next];
        let (left, top, _, bottom) = user_rect_visual_bounds(rect, geom);
        if top > current_bottom + 2.0 {
            break;
        }
        if bottom < current_top - 2.0 {
            continue;
        }
        if left <= user_rect_visual_bounds(current, geom).0 {
            continue;
        }
        obstacles.push(rect);
        if blocks[next].opaque {
            continue;
        }
        if !blocks[next].opaque {
            break;
        }
    }
    obstacles
}

fn following_inline_obstacle_x(
    index: usize,
    blocks: &[TranslatedStyledBlock],
    user_rects: &[UserRect],
    _block_styles: &[SampledBlockStyle],
    geom: PageGeometry,
) -> Option<f32> {
    following_inline_obstacles(index, blocks, user_rects, geom)
        .into_iter()
        .map(|rect| user_rect_visual_bounds(rect, geom).0)
        .reduce(f32::min)
}

/// First-line baseline y in user space. Falls back to the block bbox if
/// no anchor was sampled.
fn first_line_user_y(style: &SampledBlockStyle, user_rect: UserRect) -> f32 {
    if let Some((_, y)) = style.geometry.line_anchors.first() {
        return *y;
    }
    if let Some((_, y)) = style.geometry.anchor {
        return y;
    }
    user_rect.y1
}

/// Emit one translated block. Positioning happens in PDF user space (which
/// matches what `UserRect` carries), then we inverse-transform through the
/// page's still-active CTM into the producer's local coordinate system so
/// the appended `cm`-less stream draws at the right visual spot.
///
/// `first_line_x_override` lets the caller force the first wrapped line
/// to start at a specific x — used when this block is continuing an
/// inline row that opened with a previous opaque formula. Subsequent
/// lines fall back to their own anchors (typically the left margin).
///
/// Returns the user-space `(x, y)` where the last wrapped line ended, so
/// the caller can chain a following block onto the same visual row.
fn emit_block(
    builder: &mut ContentStreamBuilder,
    block: &TranslatedStyledBlock,
    user_rect: UserRect,
    style: &SampledBlockStyle,
    resources: &BlockResources,
    first_line_x_override: Option<f32>,
    last_line_obstacle_x: Option<f32>,
    geom: PageGeometry,
    inv_ctm: &Matrix,
) -> Option<(f32, f32)> {
    let text = block.text.trim();
    if text.is_empty() {
        return None;
    }
    let user_w = user_rect.x1 - user_rect.x0;
    let user_h = user_rect.y1 - user_rect.y0;
    if user_w <= 0.0 || user_h <= 0.0 {
        return None;
    }

    let (vis_w, vis_h) = match geom.rotate {
        90 | 270 => (user_h, user_w),
        _ => (user_w, user_h),
    };
    let line_widths = line_available_widths(
        &style.geometry,
        user_rect,
        first_line_x_override,
        last_line_obstacle_x,
        geom,
        vis_w,
    );

    let dominant_metrics = resources.dominant_metrics();
    let target_line_count = style.geometry.original_line_count;
    let (font_size, lines) = match style.typography.font_size {
        Some(size) if size.is_finite() && size > 0.0 => fit_with_sampled_size(
            text,
            &line_widths,
            vis_h,
            size,
            dominant_metrics,
            target_line_count,
        ),
        _ => {
            let initial = (vis_h * (1.0 - TEXT_BASELINE_PAD)).max(MIN_FIT_FONT_SIZE_PT);
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

    let (first_baseline_x, first_baseline_y) = match style.geometry.anchor {
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

    builder.set_fill_rgb(
        style.typography.fill_rgb.0,
        style.typography.fill_rgb.1,
        style.typography.fill_rgb.2,
    );
    builder.begin_text();

    // Map each wrapped line back to its byte ranges in `block.text` so we
    // can intersect with `block.style_spans` and produce styled segments.
    let line_word_ranges = line_byte_ranges(&block.text, &lines);

    let advance_dx = style.geometry.text_orientation.a;
    let advance_dy = style.geometry.text_orientation.b;
    let mut last_line_end: Option<(f32, f32)> = None;

    for (i, _line) in lines.iter().enumerate() {
        let (mut line_x, line_y) = line_origin(
            &style.geometry,
            i,
            first_baseline_x,
            first_baseline_y,
            leading,
            line_dx,
            line_dy,
        );
        if i == 0
            && let Some(override_x) = first_line_x_override
        {
            line_x = override_x;
        }

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
            let (seg_metrics, seg_embed) = resources.for_flags(seg.style.flags);
            let seg_resource_name: &[u8] = match seg_embed {
                Some(e) => &e.resource_name,
                None => BlockTypography::font_resource_for(FontStyleFlags {
                    bold: seg.style.flags.bold,
                    italic: seg.style.flags.italic,
                    monospace: resources.monospace,
                }),
            };
            let seg_x = line_x + cumulative * advance_dx;
            let seg_y = line_y + cumulative * advance_dy;
            let combined = Matrix {
                e: seg_x,
                f: seg_y,
                ..style.geometry.text_orientation
            };
            let tm = combined.mul(*inv_ctm);
            if let Some((r, g, b)) = seg.style.fill_rgb {
                builder.set_fill_rgb(r, g, b);
            } else {
                builder.set_fill_rgb(
                    style.typography.fill_rgb.0,
                    style.typography.fill_rgb.1,
                    style.typography.fill_rgb.2,
                );
            }
            builder.set_font(seg_resource_name, font_size);
            builder.set_text_matrix(tm);
            emit_tj_for_segment(builder, &seg.text, seg_metrics, seg_embed);

            cumulative += seg_metrics.measure(&seg.text, font_size);
        }
        last_line_end = Some((
            line_x + cumulative * advance_dx,
            line_y + cumulative * advance_dy,
        ));
    }
    builder.end_text();
    last_line_end
}

/// Re-emit a captured run of original Tjs (a preserved formula) at
/// the original positions. We deliberately keep formulas anchored where
/// they were in the source PDF: a TeX fraction renders its bar with a
/// PDF path operator (`m`, `l`, `S`) rather than a Tj, and surgery only
/// captures text-show ops. If we shifted the captured Tjs, the
/// numerator/denominator glyphs would slide one way and the fraction
/// bar would stay behind, gutting the formula. Returns the user-space
/// `(x, y)` of the formula's right edge so the caller can place the
/// next inline block (the prose continuing on the same visual row)
/// flush against it.
fn emit_captured_text_block(
    builder: &mut ContentStreamBuilder,
    captured: &[CapturedTextShow],
    user_rect: UserRect,
    style: &SampledBlockStyle,
    inv_ctm: &Matrix,
) -> Option<(f32, f32)> {
    if captured.is_empty() {
        return None;
    }

    for show in captured {
        let tm = show.combined.mul(*inv_ctm);
        builder.set_fill_rgb(show.fill_rgb.0, show.fill_rgb.1, show.fill_rgb.2);
        builder.begin_text();
        builder.set_char_spacing(show.char_spacing);
        builder.set_word_spacing(show.word_spacing);
        builder.set_horizontal_scaling(show.horizontal_scaling);
        builder.set_font(&show.font_resource, show.font_size);
        builder.set_text_matrix(tm);
        builder.push_operation(&show.operation);
        builder.end_text();
    }

    let anchor_y = style
        .geometry
        .anchor
        .map(|(_, y)| y)
        .or_else(|| captured.first().map(|s| s.origin.1))
        .unwrap_or(user_rect.y1);
    Some((user_rect.x1, anchor_y))
}

fn line_origin(
    geometry: &BlockGeometry,
    line_index: usize,
    first_x: f32,
    first_y: f32,
    leading: f32,
    line_dx: f32,
    line_dy: f32,
) -> (f32, f32) {
    if let Some(anchor) = geometry.line_anchors.get(line_index) {
        return *anchor;
    }
    if let Some(last) = geometry.line_anchors.last() {
        let extra = (line_index + 1 - geometry.line_anchors.len()) as f32 * leading;
        return (last.0 + extra * line_dx, last.1 + extra * line_dy);
    }
    let off = line_index as f32 * leading;
    (first_x + off * line_dx, first_y + off * line_dy)
}

fn line_available_widths(
    geometry: &BlockGeometry,
    user_rect: UserRect,
    first_line_x_override: Option<f32>,
    last_line_obstacle_x: Option<f32>,
    geom: PageGeometry,
    fallback: f32,
) -> Vec<f32> {
    if geometry.line_anchors.is_empty() {
        return vec![fallback.max(1.0)];
    }

    let visual_right = user_rect_visual_bounds(user_rect, geom).2;
    let last_index = geometry.line_anchors.len().saturating_sub(1);
    geometry
        .line_anchors
        .iter()
        .enumerate()
        .map(|(index, origin)| {
            let (vx, _vy) = geom.to_display(*origin);
            let effective_left = if index == 0
                && let Some(override_x) = first_line_x_override
            {
                geom.to_display((override_x, origin.1)).0
            } else {
                vx
            };
            let effective_right = if index == last_index
                && let Some(obstacle_user_x) = last_line_obstacle_x
            {
                let (ox, _) = geom.to_display((obstacle_user_x, origin.1));
                ox.min(visual_right)
            } else {
                visual_right
            };
            (effective_right - effective_left)
                .max(fallback * 0.25)
                .min(fallback)
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
        let (x, y) = geom.to_display(point);
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }
    (left, top, right, bottom)
}

/// One run of consecutive characters from a wrapped line that share the
/// same font/color style.
struct LineSegment {
    text: String,
    style: SegmentStyle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SegmentStyle {
    flags: BoldItalic,
    fill_rgb: Option<(f32, f32, f32)>,
}

/// Walk the line's words (located in `block_text` via `word_ranges`),
/// snap each alphanumeric run's style to its **majority** char flag
/// (Bergamot's token alignment often spans whitespace, so going char-by-char
/// produces off-by-one bold edges; snapping to whole words makes
/// bold/italic word-aligned). Trailing/leading punctuation inside the same
/// whitespace token (`Empleado).`) is split out into its own sub-word so
/// it picks up the punctuation's actual per-byte style instead of inheriting
/// the styled word's bold.
fn segments_for_line(
    block_text: &str,
    word_ranges: &[(usize, usize)],
    style_spans: &[StyleSpan],
    default_flags: BoldItalic,
) -> Vec<LineSegment> {
    let lookup = |byte: usize| -> SegmentStyle {
        for span in style_spans {
            if byte >= span.start as usize && byte < span.end as usize {
                if let Some(s) = &span.style {
                    return SegmentStyle {
                        flags: BoldItalic {
                            bold: s.bold,
                            italic: s.italic,
                        },
                        fill_rgb: s.text_color.map(argb_to_rgb),
                    };
                }
            }
        }
        SegmentStyle {
            flags: default_flags,
            fill_rgb: None,
        }
    };

    // Sub-word: a (start, end, separates_words) range. Each sub-word is
    // either fully alphanumeric or fully non-alphanumeric. `separates_words`
    // is true for the FIRST sub-word of a whitespace-delimited word (i.e.
    // the ones a real word-gap precedes); subsequent sub-words inside the
    // same whitespace token never get a leading space.
    let mut sub_words: Vec<(usize, usize, bool)> = Vec::new();
    for (word_start, word_end) in word_ranges {
        let mut chunk_start = *word_start;
        let mut chunk_alpha: Option<bool> = None;
        let mut byte = *word_start;
        let mut first_chunk_in_word = true;
        for c in block_text[*word_start..*word_end].chars() {
            let alpha = c.is_alphanumeric();
            if let Some(prev_alpha) = chunk_alpha
                && alpha != prev_alpha
            {
                sub_words.push((chunk_start, byte, first_chunk_in_word));
                first_chunk_in_word = false;
                chunk_start = byte;
            }
            chunk_alpha = Some(alpha);
            byte += c.len_utf8();
        }
        if chunk_alpha.is_some() {
            sub_words.push((chunk_start, byte, first_chunk_in_word));
        }
    }

    let style_for = |start: usize, end: usize, is_alpha: bool| -> SegmentStyle {
        if !is_alpha {
            return lookup(start);
        }
        let mut counts: Vec<(SegmentStyle, usize)> = Vec::new();
        let mut byte = start;
        for c in block_text[start..end].chars() {
            let style = lookup(byte);
            if let Some((_, count)) = counts.iter_mut().find(|(s, _)| *s == style) {
                *count += 1;
            } else {
                counts.push((style, 1));
            }
            byte += c.len_utf8();
        }
        counts
            .into_iter()
            .max_by_key(|(_, n)| *n)
            .map(|(s, _)| s)
            .unwrap_or(SegmentStyle {
                flags: default_flags,
                fill_rgb: None,
            })
    };

    // Group consecutive same-flag sub-words into segments. Word-separator
    // spaces stay attached to the *previous* segment so that breaking
    // segments (bold→regular transition) doesn't drop them.
    let mut segments: Vec<LineSegment> = Vec::new();
    let mut current = String::new();
    let mut current_style = SegmentStyle {
        flags: default_flags,
        fill_rgb: None,
    };
    for (i, &(start, end, separates_words)) in sub_words.iter().enumerate() {
        let chunk = &block_text[start..end];
        let is_alpha = chunk
            .chars()
            .next()
            .map(char::is_alphanumeric)
            .unwrap_or(false);
        let style = style_for(start, end, is_alpha);
        // Bergamot's SentencePiece detokenizer emits a space before
        // closing punctuation (`,`, `.`, `)`, etc.) — visually fine when
        // the surrounding text shares one font, but the bold transitions
        // we now add make the gap obvious. Suppress the separator before
        // any token that starts with a closing-punctuation glyph. Sub-words
        // that came from splitting *within* a whitespace token never get
        // a separator either.
        let hugs_previous = chunk
            .chars()
            .next()
            .is_some_and(|c| matches!(c, ',' | '.' | ')' | ']' | '}' | ':' | ';' | '?' | '!'));
        let separator = i > 0 && separates_words && !hugs_previous;
        let need_break = !current.is_empty() && style != current_style;
        if need_break {
            if separator {
                current.push(' ');
            }
            segments.push(LineSegment {
                text: std::mem::take(&mut current),
                style: current_style,
            });
            current_style = style;
        } else if separator {
            current.push(' ');
        }
        if current.is_empty() {
            current_style = style;
        }
        current.push_str(chunk);
    }
    if !current.is_empty() {
        segments.push(LineSegment {
            text: current,
            style: current_style,
        });
    }
    segments
}

fn argb_to_rgb(argb: u32) -> (f32, f32, f32) {
    (
        ((argb >> 16) & 0xFF) as f32 / 255.0,
        ((argb >> 8) & 0xFF) as f32 / 255.0,
        (argb & 0xFF) as f32 / 255.0,
    )
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
    builder: &mut ContentStreamBuilder,
    text: &str,
    metrics: &FontMetrics,
    embed: Option<&EmbeddedFont>,
) {
    if let Some(embedded) = embed {
        builder.show_hex_gids(text.chars().map(|c| {
            let original = metrics.glyph_for(c).map(|g| g.gid).unwrap_or(0);
            embedded
                .gid_remap
                .get(&original)
                .copied()
                .unwrap_or(original)
        }));
    } else {
        builder.show_winansi(text);
    }
}

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
    metrics: &FontMetrics,
    original_lines: usize,
) -> (f32, Vec<String>) {
    let min_size = (sampled * MIN_SHRINK_FRACTION).max(MIN_FIT_FONT_SIZE_PT);

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
        // re-wrap. Targeting the original line count is what actually
        // matters: the original bbox height is just the union of glyph
        // ink. Keep iterating until we fit or hit the absolute readability
        // floor — a soft per-block fraction floor is too restrictive for
        // dense pages where the substitute font (DejaVu) is wider than
        // the producer's font (CMR/CMTI) and the sampled size already
        // packs the column tight.
        let mut size = sampled;
        let mut lines = wrap_lines_to_widths(text, line_widths, size, metrics);
        let absolute_min = MIN_FIT_FONT_SIZE_PT;
        while lines.len() > original_lines && size > absolute_min {
            size = (size * MULTILINE_SHRINK_FACTOR).max(absolute_min);
            lines = wrap_lines_to_widths(text, line_widths, size, metrics);
        }
        let _ = (vis_h, min_size);
        (size, lines)
    }
}

fn wrap_to_fit(
    text: &str,
    line_widths: &[f32],
    max_height: f32,
    mut font_size: f32,
    metrics: &FontMetrics,
) -> (f32, Vec<String>) {
    for _ in 0..FIT_RETRY_LIMIT {
        let lines = wrap_lines_to_widths(text, line_widths, font_size, metrics);
        let total_height = font_size * LINE_HEIGHT_FACTOR * lines.len() as f32;
        if total_height <= max_height || font_size <= MIN_FIT_FONT_SIZE_PT {
            return (font_size, lines);
        }
        font_size *= UNWRAPPED_SHRINK_FACTOR;
    }
    let final_size = font_size.max(MIN_FIT_FONT_SIZE_PT);
    (
        final_size,
        wrap_lines_to_widths(text, line_widths, final_size, metrics),
    )
}

fn wrap_lines_to_widths(
    text: &str,
    line_widths: &[f32],
    font_size: f32,
    metrics: &FontMetrics,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for hard_line in text.split('\n') {
        let mut current = String::new();
        for word in hard_line.split_whitespace() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Body bytes between `(` and `) Tj\n` from a single show_winansi call.
    fn encode_helper(text: &str) -> Vec<u8> {
        let mut builder = ContentStreamBuilder::new();
        builder.show_winansi(text);
        let bytes = builder.finish();
        let stripped = bytes
            .strip_prefix(b"(")
            .and_then(|b| b.strip_suffix(b") Tj\n"))
            .expect("show_winansi wraps body in (...) Tj\\n");
        stripped.to_vec()
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
        let metrics = FontMetrics::approx(HELVETICA_AVG_ADVANCE);
        let lines = wrap_lines_to_widths(text, &[60.0], 10.0, &metrics);
        assert!(lines.len() > 1);
        for line in &lines {
            if line.contains(' ') {
                let w = metrics.measure(line, 10.0);
                assert!(w <= 60.0, "line too wide: {line:?} width {w}");
            }
        }
    }

    #[test]
    fn preserves_hard_line_breaks_when_wrapping() {
        let metrics = FontMetrics::approx(HELVETICA_AVG_ADVANCE);
        let lines =
            wrap_lines_to_widths("alpha beta\ngamma delta", &[200.0, 200.0], 10.0, &metrics);
        assert_eq!(lines, vec!["alpha beta", "gamma delta"]);
    }
}

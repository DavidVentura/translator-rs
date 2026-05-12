#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Rect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// A rotated rectangle. `cx`/`cy` are the centre in image pixels, `width` runs along the
/// reading direction (so for horizontal text it's the visible text width), `height` is
/// perpendicular to it, and `angle_radians` is the rotation of the reading direction relative
/// to the image's +x axis (image y points down, so a positive angle tilts text downward to the
/// right when read left-to-right).
///
/// We keep both `Rect` (AABB) and this type on detection structs. The AABB is fine for sorting,
/// hit-testing, and the Tesseract path which has no rotation information; the oriented rect is
/// what the erase/render steps consult so that tilted text (PPOCR detections) doesn't pick up
/// the inflated AABB height as its layout box.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct OrientedRect {
    pub cx: f32,
    pub cy: f32,
    pub width: f32,
    pub height: f32,
    pub angle_radians: f32,
}

impl OrientedRect {
    /// Lift an axis-aligned rectangle into an unrotated oriented rectangle. Used wherever a
    /// pipeline (e.g. Tesseract) only produces AABBs.
    pub fn axis_aligned(rect: Rect) -> Self {
        let cx = (rect.left as f32 + rect.right as f32) * 0.5;
        let cy = (rect.top as f32 + rect.bottom as f32) * 0.5;
        Self {
            cx,
            cy,
            width: rect.width() as f32,
            height: rect.height() as f32,
            angle_radians: 0.0,
        }
    }

    /// Axis-aligned bounding box of the oriented rect — the smallest `Rect` that contains all
    /// four corners. Useful for coarse hit-testing, sorting, and falling back to AABB-only
    /// downstream consumers.
    pub fn to_aabb(&self) -> Rect {
        let abs_cos = self.angle_radians.cos().abs();
        let abs_sin = self.angle_radians.sin().abs();
        let hw = self.width * 0.5;
        let hh = self.height * 0.5;
        let half_aabb_w = hw * abs_cos + hh * abs_sin;
        let half_aabb_h = hw * abs_sin + hh * abs_cos;
        Rect {
            left: (self.cx - half_aabb_w).max(0.0).round() as u32,
            top: (self.cy - half_aabb_h).max(0.0).round() as u32,
            right: (self.cx + half_aabb_w).max(0.0).round() as u32,
            bottom: (self.cy + half_aabb_h).max(0.0).round() as u32,
        }
    }

    /// Four corners in image-pixel coordinates, ordered TL, TR, BR, BL relative to the line's
    /// reading direction (so "TL" is the corner the first glyph's ascent touches, even for a
    /// rotated line).
    pub fn corners(&self) -> [(f32, f32); 4] {
        let cos = self.angle_radians.cos();
        let sin = self.angle_radians.sin();
        let hw = self.width * 0.5;
        let hh = self.height * 0.5;
        // Tangent (reading direction): (cos, sin). Perpendicular pointing across the line in
        // the +y direction of image space: (-sin, cos).
        let (tx, ty) = (cos, sin);
        let (px, py) = (-sin, cos);
        [
            (self.cx - hw * tx - hh * px, self.cy - hw * ty - hh * py),
            (self.cx + hw * tx - hh * px, self.cy + hw * ty - hh * py),
            (self.cx + hw * tx + hh * px, self.cy + hw * ty + hh * py),
            (self.cx - hw * tx + hh * px, self.cy - hw * ty + hh * py),
        ]
    }

    /// True when the oriented rect is within `epsilon` of axis-aligned; the renderer uses this
    /// to skip the rotated-rasterization path when there's nothing to rotate.
    pub fn is_axis_aligned(&self, epsilon: f32) -> bool {
        self.angle_radians.abs() < epsilon
    }
}

impl Default for OrientedRect {
    fn default() -> Self {
        Self::axis_aligned(Rect::default())
    }
}

impl Rect {
    pub fn width(&self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    pub fn height(&self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }

    pub fn center_y(&self) -> u32 {
        (self.top + self.bottom) / 2
    }

    pub fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    pub fn union(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }

        self.left = self.left.min(other.left);
        self.top = self.top.min(other.top);
        self.right = self.right.max(other.right);
        self.bottom = self.bottom.max(other.bottom);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum ReadingOrder {
    #[default]
    LeftToRight,
    TopToBottomLeftToRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
pub enum OverlayLayoutMode {
    #[default]
    PerLine,
    BlockRect,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedWord {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: Rect,
    pub is_at_beginning_of_para: bool,
    pub end_para: bool,
    pub end_line: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextLine {
    pub text: String,
    pub bounding_box: Rect,
    /// Oriented bounding box of this line. For Tesseract output and other axis-aligned
    /// pipelines this is just `bounding_box` lifted via `OrientedRect::axis_aligned`. For
    /// PPOCR it's the min-area rotated rectangle around the detection contour, so tilted
    /// text gets a tight box instead of the inflated AABB.
    pub oriented_box: OrientedRect,
    /// Glyph-tight oriented rect — same centre/angle as `oriented_box` but without
    /// ascender/descender padding (for PPOCR, the pre-unclip mask kernel). Used by paragraph
    /// grouping to cluster lines by x-height-like metric and to measure inter-line gaps in
    /// "tight heights" instead of inflated bounding-box heights. Falls back to `oriented_box`
    /// for engines that don't expose a tighter metric.
    pub tight_box: OrientedRect,
    pub word_rects: Vec<Rect>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextBlock {
    pub lines: Vec<TextLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct OverlayColors {
    pub background_argb: u32,
    pub foreground_argb: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct OverlayLayoutHints {
    pub layout_mode: OverlayLayoutMode,
    pub suggested_font_size_px: f32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct PreparedTextLine {
    pub text: String,
    pub bounding_box: Rect,
    /// Oriented box used by the overlay renderer to position rotated text. Mirror of
    /// `TextLine::oriented_box`.
    pub oriented_box: OrientedRect,
    pub word_rects: Vec<Rect>,
    pub background_argb: u32,
    pub foreground_argb: u32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct PreparedTextBlock {
    pub source_text: String,
    pub translated_text: String,
    pub bounding_box: Rect,
    pub lines: Vec<PreparedTextLine>,
    pub layout_hints: OverlayLayoutHints,
    pub background_argb: u32,
    pub foreground_argb: u32,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct PreparedImageOverlay {
    pub rgba_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub extracted_text: String,
    pub translated_text: String,
    pub blocks: Vec<PreparedTextBlock>,
}

#[derive(Debug, Clone, PartialEq)]
struct WordInfo {
    text: String,
    confidence: f32,
    bounding_box: Rect,
    ghost_bbox: Option<Rect>,
    is_first_in_line: bool,
    is_last_in_line: bool,
    is_last_in_para: bool,
}

enum LineJoin {
    /// Insert a single space between the two lines (default for Latin/Cyrillic/Greek text).
    Space,
    /// Concatenate with nothing between (CJK, or after stripping an end-of-line hyphen).
    Concat,
}

fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    // Han Unified Ideographs + Ext A + Compatibility, plus Hiragana, Katakana, and Hangul
    // syllables/Jamo. Excludes the Halfwidth/Fullwidth Forms block to avoid falsely matching
    // half-width ASCII surrogates that some OCR outputs use.
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x3040..=0x309F).contains(&cp)
        || (0x30A0..=0x30FF).contains(&cp)
        || (0xAC00..=0xD7AF).contains(&cp)
        || (0x1100..=0x11FF).contains(&cp)
}

impl TextBlock {
    pub fn source_text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn translation_text(&self) -> String {
        let mut out = String::new();
        for line in self
            .lines
            .iter()
            .map(|l| l.text.trim())
            .filter(|l| !l.is_empty())
        {
            if out.is_empty() {
                out.push_str(line);
                continue;
            }
            let prev_last = out.chars().next_back();
            let next_first = line.chars().next();
            let separator = match (prev_last, next_first) {
                // OCR-style soft hyphen at end of line preceding a lowercase letter: the source
                // word was broken at the hyphen, so glue the halves together and drop the
                // dangling hyphen. We use a permissive lowercase test to catch German/Spanish
                // diacritics, not just ASCII. Mirrors the Tesseract path's behaviour in
                // `merge_hyphenated_words`.
                (Some(prev), Some(next))
                    if matches!(prev, '-' | '\u{2010}' | '\u{00AD}') && next.is_lowercase() =>
                {
                    out.pop();
                    LineJoin::Concat
                }
                // CJK on both sides of the line break: no space.
                (Some(prev), Some(next)) if is_cjk_char(prev) && is_cjk_char(next) => {
                    LineJoin::Concat
                }
                _ => LineJoin::Space,
            };
            if matches!(separator, LineJoin::Space) {
                out.push(' ');
            }
            out.push_str(line);
        }
        out
    }

    pub fn bounds(&self) -> Rect {
        let Some(first) = self.lines.first().map(|line| line.bounding_box) else {
            return Rect::default();
        };
        let mut combined = first;
        for line in self.lines.iter().skip(1) {
            combined.union(line.bounding_box);
        }
        combined
    }
}

/// Tunables for `group_lines_into_paragraphs`. All distance tolerances are expressed in units of
/// the paragraph's running median tight-height, so the same values work across font sizes.
#[derive(Debug, Clone, Copy)]
pub struct ParagraphGroupingOptions {
    /// Max permitted relative height jitter between a new line and the paragraph's running
    /// median tight-height, once the paragraph has 2+ lines (the size has been "confirmed" by
    /// at least one accepted body→body match). `0.25` accepts lines within ±25% of the running
    /// paragraph height.
    pub height_tolerance: f32,
    /// Tighter height tolerance used when the paragraph still has only its opening line. Until
    /// we have a second body-shaped line, the "paragraph" might actually be a heading or a
    /// caption — and a slightly-shorter body line beneath a heading is still a paragraph
    /// break, not a join. Once the second line is accepted the size is "confirmed" and we
    /// switch to the looser `height_tolerance`.
    pub opening_height_tolerance: f32,
    /// Max vertical gap (top-of-new-line minus bottom-of-prev-line) in median-tight-height
    /// units before we call it a paragraph break. ~1.8 leaves room for generous leading
    /// without merging across blank lines.
    pub max_gap_ratio: f32,
    /// Max negative gap (overlap) in median-tight-height units. Allows a sliver of overlap so
    /// rasterisation jitter or `sort_lines_reading_order` bucket aliasing doesn't split
    /// paragraphs. A more-negative gap (the next "line" sits well above the previous line's
    /// bottom) is the signature of column interleaving, and should break.
    pub max_overlap_ratio: f32,
    /// Maximum left-edge jitter in median-tight-height units for two lines to count as
    /// belonging to the same column. Used both for justified text (left edges align tightly)
    /// and ragged-right (same column edge, varying right edges).
    pub edge_alignment_tolerance: f32,
    /// Maximum first-line indent in median-tight-height units. Body lines that sit further
    /// LEFT than the running `column_left` by up to this amount are accepted, and `column_left`
    /// shifts to follow them — that's how an indented first line plus flush body gets glued
    /// into one paragraph.
    pub max_first_line_indent: f32,
}

impl Default for ParagraphGroupingOptions {
    fn default() -> Self {
        Self {
            height_tolerance: 0.25,
            opening_height_tolerance: 0.15,
            max_gap_ratio: 1.8,
            max_overlap_ratio: 0.5,
            edge_alignment_tolerance: 0.6,
            max_first_line_indent: 4.0,
        }
    }
}

/// Column-tolerance for `assign_column_ids` and for the grouper's first-line-indent gate, in
/// units of median tight-height. Anything left-edge-shift smaller than this is "same column";
/// larger is a column hop. Sized so that a typical first-line indent (~2 em ≈ 3-4 × x-height)
/// stays within one column while a multi-column page's column gap (typically ≥ 8 × x-height)
/// triggers a new column.
const COLUMN_TOLERANCE_HEIGHTS: f32 = 4.0;

/// PPOCR's `sort_lines_reading_order` interleaves columns by binning on `top`, which means a
/// two-column page comes in as A0, B0, A1, B1, … — breaking the gap and alignment heuristics
/// that paragraph grouping relies on. We cluster lines into "columns" by left edge first, then
/// sort within each column top-to-bottom. The grouper then walks each column's lines in order
/// and can rely on vertical-gap / left-alignment without column-induced jitter.
///
/// Bucket-based bucketing (e.g. `left / bucket`) doesn't work here because an indented
/// first-line at `left=50` and body lines at `left=20` straddle any fixed bucket boundary near
/// 40-50 px. So we sort by left edge and grow column IDs greedily based on inter-line gaps.
/// Tight (pre-inflate) top/bottom/left edges, used for all grouping math. The AABB in
/// `TextLine::bounding_box` is unclip-and-border-inflated for PPOCR detections, so adjacent
/// lines' AABBs often overlap vertically even when the actual ink is well-separated.
/// `tight_box.cy ± height/2` and `cx - width/2` give us the actual ink extents.
fn tight_top(line: &TextLine) -> f32 {
    line.tight_box.cy - line.tight_box.height * 0.5
}
fn tight_bottom(line: &TextLine) -> f32 {
    line.tight_box.cy + line.tight_box.height * 0.5
}
fn tight_left(line: &TextLine) -> f32 {
    line.tight_box.cx - line.tight_box.width * 0.5
}

fn reorder_lines_for_grouping(mut lines: Vec<TextLine>) -> Vec<TextLine> {
    if lines.len() < 2 {
        return lines;
    }
    let mut heights: Vec<f32> = lines.iter().map(|l| l.tight_box.height.max(1.0)).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_h = heights[heights.len() / 2].max(1.0);
    let tolerance = COLUMN_TOLERANCE_HEIGHTS * median_h;
    let column_ids = assign_column_ids(&lines, tolerance);
    let mut indexed: Vec<(usize, TextLine)> = column_ids.into_iter().zip(lines).collect();
    indexed.sort_by(|(id_a, a), (id_b, b)| {
        id_a.cmp(id_b).then_with(|| {
            tight_top(a)
                .partial_cmp(&tight_top(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    indexed.into_iter().map(|(_, line)| line).collect()
}

/// Assigns each line a column ID by sorting by tight-left and growing a new ID whenever the
/// gap between two consecutive tight-left edges exceeds `tolerance` pixels. The walk is in
/// sort order, so a staircase of left edges within tolerance steps still ends up as a single
/// column — this is what lets an indented first-line stay in the same column as its flush
/// body lines.
fn assign_column_ids(lines: &[TextLine], tolerance: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by(|&a, &b| {
        tight_left(&lines[a])
            .partial_cmp(&tight_left(&lines[b]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ids = vec![0usize; lines.len()];
    let mut current_id = 0usize;
    let mut prev_left: Option<f32> = None;
    for &i in &order {
        let left = tight_left(&lines[i]);
        if let Some(p) = prev_left
            && (left - p) > tolerance
        {
            current_id += 1;
        }
        ids[i] = current_id;
        prev_left = Some(left);
    }
    ids
}

struct ParaState {
    lines: Vec<TextLine>,
    /// Running `min(left)` across the paragraph's lines. For both justified and ragged-right
    /// blocks, every line's left edge sits at this value (modulo first-line indent which
    /// drives the min downward when the body line arrives). This single value lets us reject
    /// new-column lines (large positive delta) while still admitting indented first lines
    /// (negative delta within `max_first_line_indent`).
    column_left: f32,
    /// Running estimate of the paragraph's tight-height. Updated as a simple EMA so the gate
    /// drifts with the paragraph but isn't pinned to the first line.
    median_h: f32,
}

/// Group already-line-detected text into paragraphs.
///
/// Input is expected to be in reading order (top-to-bottom, left-to-right within row) — for the
/// PPOCR pipeline that's what `sort_lines_reading_order` produces. Each `TextLine` should carry
/// its glyph-tight `tight_box` (PPOCR's pre-unclip mask rect); engines that don't expose one
/// can pass `oriented_box` as the tight box at the cost of slightly looser gap/height checks
/// because of ascender/descender padding.
///
/// V1 grouping criteria (all must pass to merge a new line into the running paragraph):
///   * **Height match.** New line's tight-height within `±height_tolerance` of running median.
///   * **Vertical gap.** Baseline-to-baseline gap within `[-max_overlap_ratio, max_gap_ratio]`
///     median-tight-height units. Stops column-interleaved sort artifacts (large negative gap)
///     and blank-line paragraph breaks (large positive gap).
///   * **Column alignment.** New line's left edge within `edge_alignment_tolerance` of running
///     `column_left`, OR up to `max_first_line_indent` units to the left of it (in which case
///     `column_left` shifts to follow the new line — handles first-line indent).
///
/// We don't gate on the oriented-rect angle in V1 — perspective shift and DB-mask jitter make
/// the per-line angle estimate too unstable to trust as a paragraph-break signal.
///
/// Right-justified and centered blocks aren't detected as multi-line paragraphs here; their
/// lines will fall out as one-line paragraphs (same as the pre-change behaviour). That's a
/// deliberate scope cut.
pub fn group_lines_into_paragraphs(
    lines: Vec<TextLine>,
    opts: ParagraphGroupingOptions,
) -> Vec<TextBlock> {
    let input_count = lines.len();
    let lines = reorder_lines_for_grouping(lines);
    let mut paragraphs: Vec<TextBlock> = Vec::new();
    let mut current: Option<ParaState> = None;

    for line in lines {
        let h = line.tight_box.height.max(1.0);
        let top = tight_top(&line);
        let left = tight_left(&line);

        let decision = match current.as_ref() {
            None => JoinDecision::OpenFirst,
            Some(state) => {
                let unit = state.median_h.max(1.0);
                let last = state
                    .lines
                    .last()
                    .expect("paragraph state always has at least one line");
                let last_bottom = tight_bottom(last);

                let big_h = h.max(state.median_h);
                let small_h = h.min(state.median_h).max(1.0);
                let height_ratio_excess = big_h / small_h - 1.0;
                // Use the tighter "opening" tolerance until the paragraph's height has been
                // confirmed by at least one accepted line. After that, real per-line OCR
                // jitter can be a touch larger than the heading/body gap we want to catch,
                // so we relax to `height_tolerance`.
                let active_height_tolerance = if state.lines.len() == 1 {
                    opts.opening_height_tolerance
                } else {
                    opts.height_tolerance
                };
                let height_ok = height_ratio_excess <= active_height_tolerance;

                let gap = top - last_bottom;
                let gap_ratio = gap / unit;
                let gap_ok = gap >= -opts.max_overlap_ratio * unit
                    && gap.max(0.0) <= opts.max_gap_ratio * unit;

                let delta = left - state.column_left;
                let abs_delta = delta.abs();
                let aligned = abs_delta <= opts.edge_alignment_tolerance * unit;
                let indent_shift = delta < 0.0 && (-delta) <= opts.max_first_line_indent * unit;
                let align_ok = aligned || indent_shift;

                if height_ok && gap_ok && align_ok {
                    JoinDecision::Join
                } else {
                    JoinDecision::Break {
                        height_excess: height_ratio_excess,
                        height_limit: active_height_tolerance,
                        gap_ratio,
                        delta,
                        unit,
                    }
                }
            }
        };

        match decision {
            JoinDecision::Join => {
                let state = current
                    .as_mut()
                    .expect("join implies a current paragraph exists");
                state.column_left = state.column_left.min(left);
                // EMA towards each new line's tight-height. Reacts to drift over a few lines
                // without letting a single outlier dominate.
                state.median_h = state.median_h * 0.7 + h * 0.3;
                state.lines.push(line);
            }
            JoinDecision::OpenFirst | JoinDecision::Break { .. } => {
                if let JoinDecision::Break {
                    height_excess,
                    height_limit,
                    gap_ratio,
                    delta,
                    unit,
                } = decision
                {
                    log::debug!(
                        "ppocr group break: \"{}\" h={:.1} top={:.1} left={:.1} \
                         vs prev unit={:.1} → height_excess={:.2} (limit {:.2}) \
                         gap_ratio={:.2} (overlap {:.2}, gap {:.2}) \
                         delta_left={:.1} (align {:.1}, indent {:.1})",
                        truncate_for_log(&line.text),
                        h,
                        top,
                        left,
                        unit,
                        height_excess,
                        height_limit,
                        gap_ratio,
                        -opts.max_overlap_ratio,
                        opts.max_gap_ratio,
                        delta,
                        opts.edge_alignment_tolerance * unit,
                        opts.max_first_line_indent * unit,
                    );
                }
                if let Some(state) = current.take() {
                    paragraphs.push(TextBlock { lines: state.lines });
                }
                current = Some(ParaState {
                    lines: vec![line],
                    column_left: left,
                    median_h: h,
                });
            }
        }
    }

    if let Some(state) = current.take() {
        paragraphs.push(TextBlock { lines: state.lines });
    }

    log::info!(
        "ppocr paragraph grouping: {} lines → {} paragraph(s)",
        input_count,
        paragraphs.len(),
    );

    paragraphs
}

enum JoinDecision {
    OpenFirst,
    Join,
    Break {
        height_excess: f32,
        height_limit: f32,
        gap_ratio: f32,
        delta: f32,
        unit: f32,
    },
}

fn truncate_for_log(text: &str) -> String {
    const MAX_CHARS: usize = 40;
    let mut out = String::with_capacity(MAX_CHARS + 3);
    for (i, c) in text.chars().enumerate() {
        if i >= MAX_CHARS {
            out.push('…');
            break;
        }
        out.push(c);
    }
    out
}

fn overlay_layout_hints(block: &TextBlock, reading_order: ReadingOrder) -> OverlayLayoutHints {
    let layout_mode = match reading_order {
        ReadingOrder::LeftToRight => OverlayLayoutMode::PerLine,
        ReadingOrder::TopToBottomLeftToRight => OverlayLayoutMode::BlockRect,
    };
    let suggested_font_size_px = if block.lines.is_empty() {
        match reading_order {
            ReadingOrder::LeftToRight => block.bounds().height() as f32,
            ReadingOrder::TopToBottomLeftToRight => block.bounds().width() as f32,
        }
    } else {
        // For per-line layout, use the oriented box's height (perpendicular to reading
        // direction) so tilted lines aren't sized off their inflated AABB height. For block-
        // rect (top-to-bottom) layout the box is axis-aligned, so the oriented and AABB widths
        // are equal.
        let total = block
            .lines
            .iter()
            .map(|line| match reading_order {
                ReadingOrder::LeftToRight => line.oriented_box.height,
                ReadingOrder::TopToBottomLeftToRight => line.bounding_box.width() as f32,
            })
            .sum::<f32>();
        total / block.lines.len() as f32
    };
    OverlayLayoutHints {
        layout_mode,
        suggested_font_size_px,
    }
}

struct RasterImage<'a> {
    width: u32,
    height: u32,
    rgba: &'a [u8],
}

struct RasterImageMut {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl<'a> RasterImage<'a> {
    fn new(rgba: &'a [u8], width: u32, height: u32) -> Result<Self, String> {
        let expected_len = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "image dimensions overflow".to_string())?
            as usize;
        if rgba.len() != expected_len {
            return Err(format!(
                "invalid rgba size: expected {expected_len}, got {}",
                rgba.len()
            ));
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    fn pixel_argb(&self, x: u32, y: u32) -> u32 {
        let index = ((y * self.width + x) * 4) as usize;
        u32::from_ne_bytes([
            self.rgba[index],
            self.rgba[index + 1],
            self.rgba[index + 2],
            self.rgba[index + 3],
        ])
    }
}

impl RasterImageMut {
    fn new(rgba: &[u8], width: u32, height: u32) -> Result<Self, String> {
        let image = RasterImage::new(rgba, width, height)?;
        Ok(Self {
            width: image.width,
            height: image.height,
            rgba: rgba.to_vec(),
        })
    }

    fn as_image(&self) -> RasterImage<'_> {
        RasterImage {
            width: self.width,
            height: self.height,
            rgba: &self.rgba,
        }
    }

    fn fill_rect(&mut self, rect: Rect, argb: u32) {
        let Some(rect) = clamp_rect(rect, self.width, self.height) else {
            return;
        };
        let bytes = argb.to_ne_bytes();
        for y in rect.top..rect.bottom {
            for x in rect.left..rect.right {
                let index = ((y * self.width + x) * 4) as usize;
                self.rgba[index..index + 4].copy_from_slice(&bytes);
            }
        }
    }

    fn fill_bilinear(&mut self, rect: Rect, tl: u32, tr: u32, bl: u32, br: u32) {
        let Some(rect) = clamp_rect(rect, self.width, self.height) else {
            return;
        };
        let w = rect.width();
        let h = rect.height();
        if w == 0 || h == 0 {
            return;
        }
        let max_u = (w.saturating_sub(1).max(1)) as f32;
        let max_v = (h.saturating_sub(1).max(1)) as f32;
        let rgb = |c: u32| -> [f32; 3] {
            [
                channel_r(c) as f32,
                channel_g(c) as f32,
                channel_b(c) as f32,
            ]
        };
        let tl_c = rgb(tl);
        let tr_c = rgb(tr);
        let bl_c = rgb(bl);
        let br_c = rgb(br);

        for y in rect.top..rect.bottom {
            let v = (y - rect.top) as f32 / max_v;
            let left = [
                tl_c[0] + (bl_c[0] - tl_c[0]) * v,
                tl_c[1] + (bl_c[1] - tl_c[1]) * v,
                tl_c[2] + (bl_c[2] - tl_c[2]) * v,
            ];
            let right = [
                tr_c[0] + (br_c[0] - tr_c[0]) * v,
                tr_c[1] + (br_c[1] - tr_c[1]) * v,
                tr_c[2] + (br_c[2] - tr_c[2]) * v,
            ];
            for x in rect.left..rect.right {
                let u = (x - rect.left) as f32 / max_u;
                let r = (left[0] + (right[0] - left[0]) * u).clamp(0.0, 255.0) as u8;
                let g = (left[1] + (right[1] - left[1]) * u).clamp(0.0, 255.0) as u8;
                let b = (left[2] + (right[2] - left[2]) * u).clamp(0.0, 255.0) as u8;
                let bytes = argb(r, g, b).to_ne_bytes();
                let idx = ((y * self.width + x) * 4) as usize;
                self.rgba[idx..idx + 4].copy_from_slice(&bytes);
            }
        }
    }

    fn apply_fill_plan(&mut self, rect: Rect, plan: FillPlan) {
        match plan {
            FillPlan::Flat(color) => self.fill_rect(rect, color),
            FillPlan::Bilinear { tl, tr, bl, br } => self.fill_bilinear(rect, tl, tr, bl, br),
        }
    }

    /// Fill a rotated rectangle with a flat color. Iterates over the rect's AABB and accepts
    /// pixels whose line-local coordinates fall inside the half-extents. Falls back to
    /// `fill_rect` when the rect is effectively axis-aligned so the unrotated path stays
    /// pixel-exact.
    fn fill_oriented_rect(&mut self, rect: OrientedRect, argb: u32) {
        if rect.is_axis_aligned(AXIS_ALIGNED_EPSILON_RAD) {
            self.fill_rect(rect.to_aabb(), argb);
            return;
        }
        let cos = rect.angle_radians.cos();
        let sin = rect.angle_radians.sin();
        let Some(aabb) = clamp_rect(rect.to_aabb(), self.width, self.height) else {
            return;
        };
        let hw = rect.width * 0.5;
        let hh = rect.height * 0.5;
        let bytes = argb.to_ne_bytes();
        for y in aabb.top..aabb.bottom {
            for x in aabb.left..aabb.right {
                let dx = x as f32 + 0.5 - rect.cx;
                let dy = y as f32 + 0.5 - rect.cy;
                let lx = dx * cos + dy * sin;
                let ly = -dx * sin + dy * cos;
                if lx.abs() <= hw && ly.abs() <= hh {
                    let index = ((y * self.width + x) * 4) as usize;
                    self.rgba[index..index + 4].copy_from_slice(&bytes);
                }
            }
        }
    }

    /// Bilinear corner-color gradient in line-local (u, v) ∈ [0,1] space inside the rotated
    /// rectangle. Mirrors `fill_bilinear`'s semantics so AutoDetect's gradient erase works
    /// equally well on tilted lines.
    fn fill_oriented_bilinear(&mut self, rect: OrientedRect, tl: u32, tr: u32, bl: u32, br: u32) {
        if rect.is_axis_aligned(AXIS_ALIGNED_EPSILON_RAD) {
            self.fill_bilinear(rect.to_aabb(), tl, tr, bl, br);
            return;
        }
        if rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }
        let cos = rect.angle_radians.cos();
        let sin = rect.angle_radians.sin();
        let Some(aabb) = clamp_rect(rect.to_aabb(), self.width, self.height) else {
            return;
        };
        let hw = rect.width * 0.5;
        let hh = rect.height * 0.5;
        let rgb = |c: u32| -> [f32; 3] {
            [
                channel_r(c) as f32,
                channel_g(c) as f32,
                channel_b(c) as f32,
            ]
        };
        let tl_c = rgb(tl);
        let tr_c = rgb(tr);
        let bl_c = rgb(bl);
        let br_c = rgb(br);
        for y in aabb.top..aabb.bottom {
            for x in aabb.left..aabb.right {
                let dx = x as f32 + 0.5 - rect.cx;
                let dy = y as f32 + 0.5 - rect.cy;
                let lx = dx * cos + dy * sin;
                let ly = -dx * sin + dy * cos;
                if lx.abs() > hw || ly.abs() > hh {
                    continue;
                }
                let u = (lx + hw) / rect.width;
                let v = (ly + hh) / rect.height;
                let left = [
                    tl_c[0] + (bl_c[0] - tl_c[0]) * v,
                    tl_c[1] + (bl_c[1] - tl_c[1]) * v,
                    tl_c[2] + (bl_c[2] - tl_c[2]) * v,
                ];
                let right = [
                    tr_c[0] + (br_c[0] - tr_c[0]) * v,
                    tr_c[1] + (br_c[1] - tr_c[1]) * v,
                    tr_c[2] + (br_c[2] - tr_c[2]) * v,
                ];
                let r = (left[0] + (right[0] - left[0]) * u).clamp(0.0, 255.0) as u8;
                let g = (left[1] + (right[1] - left[1]) * u).clamp(0.0, 255.0) as u8;
                let b = (left[2] + (right[2] - left[2]) * u).clamp(0.0, 255.0) as u8;
                let bytes = argb(r, g, b).to_ne_bytes();
                let idx = ((y * self.width + x) * 4) as usize;
                self.rgba[idx..idx + 4].copy_from_slice(&bytes);
            }
        }
    }

    fn apply_fill_plan_oriented(&mut self, rect: OrientedRect, plan: FillPlan) {
        match plan {
            FillPlan::Flat(color) => self.fill_oriented_rect(rect, color),
            FillPlan::Bilinear { tl, tr, bl, br } => {
                self.fill_oriented_bilinear(rect, tl, tr, bl, br)
            }
        }
    }
}

/// Within ~0.057° of horizontal we treat the rect as axis-aligned and use the pixel-exact
/// `fill_rect` path; below this threshold the rotated rasterization's sub-pixel error is
/// larger than the rotation itself.
const AXIS_ALIGNED_EPSILON_RAD: f32 = 0.001;

fn channel_r(color: u32) -> u8 {
    ((color >> 16) & 0xFF) as u8
}

fn channel_g(color: u32) -> u8 {
    ((color >> 8) & 0xFF) as u8
}

fn channel_b(color: u32) -> u8 {
    (color & 0xFF) as u8
}

fn argb(r: u8, g: u8, b: u8) -> u32 {
    0xFF00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn clamp_rect(rect: Rect, width: u32, height: u32) -> Option<Rect> {
    if width == 0 || height == 0 {
        return None;
    }
    let left = rect.left.min(width - 1);
    let top = rect.top.min(height - 1);
    let right = rect.right.clamp(left + 1, width);
    let bottom = rect.bottom.clamp(top + 1, height);
    let clamped = Rect {
        left,
        top,
        right,
        bottom,
    };
    if clamped.is_empty() {
        None
    } else {
        Some(clamped)
    }
}

pub fn luminance(color: u32) -> f32 {
    let r = channel_r(color) as f32 / 255.0;
    let g = channel_g(color) as f32 / 255.0;
    let b = channel_b(color) as f32 / 255.0;
    0.299 * r + 0.587 * g + 0.114 * b
}

fn luminance_u8(color: u32) -> u8 {
    let r = channel_r(color) as u32;
    let g = channel_g(color) as u32;
    let b = channel_b(color) as u32;
    ((77 * r + 150 * g + 29 * b) >> 8).min(255) as u8
}

fn get_surrounding_average_color(image: &RasterImage<'_>, text_bounds: Rect) -> u32 {
    let margin = 4;
    let sample_regions = [
        Rect {
            left: text_bounds.left.saturating_sub(margin),
            top: text_bounds.top,
            right: text_bounds.left,
            bottom: text_bounds.bottom,
        },
        Rect {
            left: text_bounds.right,
            top: text_bounds.top,
            right: (text_bounds.right + margin).min(image.width),
            bottom: text_bounds.bottom,
        },
        Rect {
            left: text_bounds.left,
            top: text_bounds.top.saturating_sub(margin),
            right: text_bounds.right,
            bottom: text_bounds.top,
        },
        Rect {
            left: text_bounds.left,
            top: text_bounds.bottom,
            right: text_bounds.right,
            bottom: (text_bounds.bottom + margin).min(image.height),
        },
    ];

    let mut total_r = 0u64;
    let mut total_g = 0u64;
    let mut total_b = 0u64;
    let mut count = 0u64;

    for region in sample_regions {
        let Some(region) = clamp_rect(region, image.width, image.height) else {
            continue;
        };
        for y in region.top..region.bottom {
            for x in region.left..region.right {
                let pixel = image.pixel_argb(x, y);
                total_r += channel_r(pixel) as u64;
                total_g += channel_g(pixel) as u64;
                total_b += channel_b(pixel) as u64;
                count += 1;
            }
        }
    }

    if count == 0 {
        argb(255, 255, 255)
    } else {
        argb(
            (total_r / count) as u8,
            (total_g / count) as u8,
            (total_b / count) as u8,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillPlan {
    Flat(u32),
    Bilinear { tl: u32, tr: u32, bl: u32, br: u32 },
}

struct AutoDetectPaint {
    fill: FillPlan,
    colors: OverlayColors,
}

// Corner colors are considered uniform enough for a flat fill when every
// channel is within this many units of the 4-corner average.
const FLAT_FILL_DELTA: u32 = 4;

fn otsu_threshold(histogram: &[u64; 256]) -> u8 {
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return 127;
    }
    let total_sum: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();

    let mut w_bg = 0u64;
    let mut sum_bg = 0f64;
    let mut best_variance = -1f64;
    let mut best_threshold = 127u8;
    for t in 0..256 {
        w_bg += histogram[t];
        if w_bg == 0 {
            continue;
        }
        let w_fg = total - w_bg;
        if w_fg == 0 {
            break;
        }
        sum_bg += t as f64 * histogram[t] as f64;
        let mean_bg = sum_bg / w_bg as f64;
        let mean_fg = (total_sum - sum_bg) / w_fg as f64;
        let variance = w_bg as f64 * w_fg as f64 * (mean_bg - mean_fg).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_threshold = t as u8;
        }
    }
    best_threshold
}

fn mean_color(r: u64, g: u64, b: u64, n: u64) -> Option<u32> {
    if n == 0 {
        None
    } else {
        Some(argb((r / n) as u8, (g / n) as u8, (b / n) as u8))
    }
}

fn average_corner_color(corners: [u32; 4]) -> u32 {
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;
    for c in corners {
        r += channel_r(c) as u32;
        g += channel_g(c) as u32;
        b += channel_b(c) as u32;
    }
    argb((r / 4) as u8, (g / 4) as u8, (b / 4) as u8)
}

fn max_channel_delta(a: u32, b: u32) -> u32 {
    let dr = (channel_r(a) as i32 - channel_r(b) as i32).unsigned_abs();
    let dg = (channel_g(a) as i32 - channel_g(b) as i32).unsigned_abs();
    let db = (channel_b(a) as i32 - channel_b(b) as i32).unsigned_abs();
    dr.max(dg).max(db)
}

fn peak_luminance(histogram: &[u64; 256], low: usize, high_exclusive: usize) -> u8 {
    let mut best_count = 0u64;
    let mut best = ((low + high_exclusive.saturating_sub(1)) / 2) as u8;
    for i in low..high_exclusive {
        if histogram[i] > best_count {
            best_count = histogram[i];
            best = i as u8;
        }
    }
    best
}

fn autodetect_paint(image: &RasterImage<'_>, bounds: Rect) -> AutoDetectPaint {
    let fallback = |bg: u32| {
        let fg = if luminance(bg) > 0.5 {
            argb(0, 0, 0)
        } else {
            argb(255, 255, 255)
        };
        AutoDetectPaint {
            fill: FillPlan::Flat(bg),
            colors: OverlayColors {
                background_argb: bg,
                foreground_argb: fg,
            },
        }
    };

    let Some(bounds) = clamp_rect(bounds, image.width, image.height) else {
        return fallback(argb(255, 255, 255));
    };
    if bounds.width() < 2 || bounds.height() < 2 {
        return fallback(get_surrounding_average_color(image, bounds));
    }

    let mut histogram = [0u64; 256];
    for y in bounds.top..bounds.bottom {
        for x in bounds.left..bounds.right {
            let pixel = image.pixel_argb(x, y);
            histogram[luminance_u8(pixel) as usize] += 1;
        }
    }
    let threshold = otsu_threshold(&histogram);

    let surround = get_surrounding_average_color(image, bounds);
    let bg_is_bright = luminance_u8(surround) > threshold;

    // BG = pixels within a narrow band around the paper's modal luminance.
    // Rejects specular highlights (too bright) and anti-aliased stroke
    // transitions (near threshold).
    let (bg_lo, bg_hi) = if bg_is_bright {
        (threshold as usize + 1, 256)
    } else {
        (0, threshold as usize + 1)
    };
    let paper_peak = peak_luminance(&histogram, bg_lo, bg_hi);
    const BG_BAND: u8 = 12;

    // FG = extreme 5% of fg-cluster by luminance. This captures core-ink
    // even on low-contrast lines where most fg pixels are anti-aliased
    // greys (the peak of the fg cluster can sit in the anti-aliased band).
    let fg_total_count: u64 = if bg_is_bright {
        histogram[0..=threshold as usize].iter().sum()
    } else {
        histogram[threshold as usize + 1..256].iter().sum()
    };
    let fg_percentile_target = (fg_total_count / 20).max(1);
    let fg_cutoff: u8 = if bg_is_bright {
        let mut cumulative = 0u64;
        let mut cutoff = threshold;
        for i in 0..=threshold as usize {
            cumulative += histogram[i];
            if cumulative >= fg_percentile_target {
                cutoff = i as u8;
                break;
            }
        }
        cutoff
    } else {
        let mut cumulative = 0u64;
        let mut cutoff = threshold.saturating_add(1);
        for i in (threshold as usize + 1..256).rev() {
            cumulative += histogram[i];
            if cumulative >= fg_percentile_target {
                cutoff = i as u8;
                break;
            }
        }
        cutoff
    };

    #[derive(Default, Clone, Copy)]
    struct Accum {
        r: u64,
        g: u64,
        b: u64,
        n: u64,
    }
    let mut bg_quad = [Accum::default(); 4];
    let mut fg_total = Accum::default();

    // FG is sampled from the interior — that's where the ink lives.
    for y in bounds.top..bounds.bottom {
        for x in bounds.left..bounds.right {
            let pixel = image.pixel_argb(x, y);
            let lum = luminance_u8(pixel);
            let is_core_fg = if bg_is_bright {
                lum <= fg_cutoff
            } else {
                lum >= fg_cutoff
            };
            if is_core_fg {
                fg_total.r += channel_r(pixel) as u64;
                fg_total.g += channel_g(pixel) as u64;
                fg_total.b += channel_b(pixel) as u64;
                fg_total.n += 1;
            }
        }
    }

    // BG = per-corner sample from a small L-shaped window of paper immediately
    // around each corner. Keeping each sample corner-local means the fill
    // matches the paper at that exact edge — important when the page has
    // horizontal gradients or lighting bands that vary across the text column.
    const SURROUND_MARGIN: u32 = 6;
    const CORNER_EXTENT: u32 = 8;
    let add_bg = |bg_quad: &mut [Accum; 4], pixel: u32, qi: usize| {
        let lum = luminance_u8(pixel);
        if lum.abs_diff(paper_peak) > BG_BAND {
            return;
        }
        let a = &mut bg_quad[qi];
        a.r += channel_r(pixel) as u64;
        a.g += channel_g(pixel) as u64;
        a.b += channel_b(pixel) as u64;
        a.n += 1;
    };
    let windows: [(u32, u32, u32, u32); 4] = [
        // tl: x in [left-margin, left+extent], y in [top-margin, top+extent]
        (
            bounds.left.saturating_sub(SURROUND_MARGIN),
            (bounds.left + CORNER_EXTENT).min(bounds.right),
            bounds.top.saturating_sub(SURROUND_MARGIN),
            (bounds.top + CORNER_EXTENT).min(bounds.bottom),
        ),
        // tr
        (
            bounds.right.saturating_sub(CORNER_EXTENT).max(bounds.left),
            (bounds.right + SURROUND_MARGIN).min(image.width),
            bounds.top.saturating_sub(SURROUND_MARGIN),
            (bounds.top + CORNER_EXTENT).min(bounds.bottom),
        ),
        // bl
        (
            bounds.left.saturating_sub(SURROUND_MARGIN),
            (bounds.left + CORNER_EXTENT).min(bounds.right),
            bounds.bottom.saturating_sub(CORNER_EXTENT).max(bounds.top),
            (bounds.bottom + SURROUND_MARGIN).min(image.height),
        ),
        // br
        (
            bounds.right.saturating_sub(CORNER_EXTENT).max(bounds.left),
            (bounds.right + SURROUND_MARGIN).min(image.width),
            bounds.bottom.saturating_sub(CORNER_EXTENT).max(bounds.top),
            (bounds.bottom + SURROUND_MARGIN).min(image.height),
        ),
    ];

    for (qi, (x0, x1, y0, y1)) in windows.iter().copied().enumerate() {
        for y in y0..y1 {
            for x in x0..x1 {
                // Skip the text-rect interior — those pixels carry ink.
                if x >= bounds.left && x < bounds.right && y >= bounds.top && y < bounds.bottom {
                    continue;
                }
                add_bg(&mut bg_quad, image.pixel_argb(x, y), qi);
            }
        }
    }

    let quad_color = |q: &Accum| mean_color(q.r, q.g, q.b, q.n).unwrap_or(surround);
    let corners = [
        quad_color(&bg_quad[0]),
        quad_color(&bg_quad[1]),
        quad_color(&bg_quad[2]),
        quad_color(&bg_quad[3]),
    ];
    let foreground_argb = mean_color(fg_total.r, fg_total.g, fg_total.b, fg_total.n).unwrap_or({
        if bg_is_bright {
            argb(0, 0, 0)
        } else {
            argb(255, 255, 255)
        }
    });

    let avg_bg = average_corner_color(corners);
    let max_delta = corners
        .iter()
        .map(|&c| max_channel_delta(c, avg_bg))
        .max()
        .unwrap_or(0);
    let fill = if max_delta <= FLAT_FILL_DELTA {
        FillPlan::Flat(avg_bg)
    } else {
        FillPlan::Bilinear {
            tl: corners[0],
            tr: corners[1],
            bl: corners[2],
            br: corners[3],
        }
    };

    AutoDetectPaint {
        fill,
        colors: OverlayColors {
            background_argb: avg_bg,
            foreground_argb,
        },
    }
}

fn get_overlay_colors(
    image: &RasterImage<'_>,
    bounds: Rect,
    background_mode: crate::BackgroundMode,
) -> OverlayColors {
    match background_mode {
        crate::BackgroundMode::WhiteOnBlack => OverlayColors {
            background_argb: argb(0, 0, 0),
            foreground_argb: argb(255, 255, 255),
        },
        crate::BackgroundMode::BlackOnWhite => OverlayColors {
            background_argb: argb(255, 255, 255),
            foreground_argb: argb(0, 0, 0),
        },
        crate::BackgroundMode::AutoDetect => autodetect_paint(image, bounds).colors,
    }
}

pub fn sample_overlay_colors(
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    bounds: Rect,
    background_mode: crate::BackgroundMode,
    _word_rects: Option<&[Rect]>,
) -> Result<OverlayColors, String> {
    let image = RasterImage::new(rgba_bytes, width, height)?;
    Ok(get_overlay_colors(&image, bounds, background_mode))
}

fn erase_text_region(
    image: &mut RasterImageMut,
    oriented: OrientedRect,
    background_mode: crate::BackgroundMode,
) -> OverlayColors {
    // The oriented rect carries the same DB-unclip + DET_BOX_BORDER inflation the AABB path
    // applies (see `oriented_rect_from_contour`), so it reliably covers ascenders/descenders
    // without spilling sideways the way an AABB does for tilted lines. AutoDetect samples
    // surrounding colours via the rect's AABB; that's only used to pick a colour, not for the
    // erase shape itself.
    let aabb = oriented.to_aabb();
    match background_mode {
        crate::BackgroundMode::WhiteOnBlack => {
            let colors = OverlayColors {
                background_argb: argb(0, 0, 0),
                foreground_argb: argb(255, 255, 255),
            };
            image.fill_oriented_rect(oriented, colors.background_argb);
            colors
        }
        crate::BackgroundMode::BlackOnWhite => {
            let colors = OverlayColors {
                background_argb: argb(255, 255, 255),
                foreground_argb: argb(0, 0, 0),
            };
            image.fill_oriented_rect(oriented, colors.background_argb);
            colors
        }
        crate::BackgroundMode::AutoDetect => {
            let paint = autodetect_paint(&image.as_image(), aabb);
            image.apply_fill_plan_oriented(oriented, paint.fill);
            paint.colors
        }
    }
}

pub fn prepare_overlay_image(
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    blocks: &[TextBlock],
    translated_blocks: &[String],
    background_mode: crate::BackgroundMode,
    reading_order: ReadingOrder,
) -> Result<PreparedImageOverlay, String> {
    let mut image = RasterImageMut::new(rgba_bytes, width, height)?;
    let mut prepared_blocks = Vec::with_capacity(blocks.len());

    for (block, translated_text) in blocks.iter().zip(translated_blocks.iter()) {
        let block_bounds = block.bounds();
        let layout_hints = overlay_layout_hints(block, reading_order);
        match reading_order {
            ReadingOrder::LeftToRight => {
                let mut prepared_lines = Vec::with_capacity(block.lines.len());
                let mut block_background = argb(255, 255, 255);
                let mut block_foreground = argb(0, 0, 0);
                for (index, line) in block.lines.iter().enumerate() {
                    let colors = erase_text_region(&mut image, line.oriented_box, background_mode);
                    if index == 0 {
                        block_background = colors.background_argb;
                        block_foreground = colors.foreground_argb;
                    }
                    prepared_lines.push(PreparedTextLine {
                        text: line.text.clone(),
                        bounding_box: line.bounding_box,
                        oriented_box: line.oriented_box,
                        word_rects: line.word_rects.clone(),
                        background_argb: colors.background_argb,
                        foreground_argb: colors.foreground_argb,
                    });
                }
                prepared_blocks.push(PreparedTextBlock {
                    source_text: block.source_text(),
                    translated_text: translated_text.clone(),
                    bounding_box: block_bounds,
                    lines: prepared_lines,
                    layout_hints,
                    background_argb: block_background,
                    foreground_argb: block_foreground,
                });
            }
            ReadingOrder::TopToBottomLeftToRight => {
                // Block-rect (CJK vertical) layout: the per-block region is the union of
                // possibly differently-rotated lines, so rotation doesn't carry up. Erase the
                // block AABB unrotated.
                let colors = erase_text_region(
                    &mut image,
                    OrientedRect::axis_aligned(block_bounds),
                    background_mode,
                );
                let prepared_lines = block
                    .lines
                    .iter()
                    .map(|line| PreparedTextLine {
                        text: line.text.clone(),
                        bounding_box: line.bounding_box,
                        oriented_box: line.oriented_box,
                        word_rects: line.word_rects.clone(),
                        background_argb: colors.background_argb,
                        foreground_argb: colors.foreground_argb,
                    })
                    .collect();
                prepared_blocks.push(PreparedTextBlock {
                    source_text: block.source_text(),
                    translated_text: translated_text.clone(),
                    bounding_box: block_bounds,
                    lines: prepared_lines,
                    layout_hints,
                    background_argb: colors.background_argb,
                    foreground_argb: colors.foreground_argb,
                });
            }
        }
    }

    Ok(PreparedImageOverlay {
        rgba_bytes: image.rgba,
        width,
        height,
        extracted_text: blocks
            .iter()
            .map(TextBlock::source_text)
            .collect::<Vec<_>>()
            .join("\n"),
        translated_text: translated_blocks.join("\n"),
        blocks: prepared_blocks,
    })
}

fn merge_hyphenated_words(words: Vec<WordInfo>) -> Vec<WordInfo> {
    if words.is_empty() {
        return words;
    }

    let mut result = Vec::new();
    let mut index = 0;

    while index < words.len() {
        let current_word = &words[index];
        if index == words.len() - 1 {
            result.push(current_word.clone());
            break;
        }

        if !current_word.is_last_in_line || !current_word.text.ends_with('-') {
            result.push(current_word.clone());
            index += 1;
            continue;
        }

        let next_word = &words[index + 1];
        let poor_mans_first_in_line = next_word.bounding_box.left < current_word.bounding_box.left
            && next_word.bounding_box.top > current_word.bounding_box.top;
        if !next_word.is_first_in_line && !poor_mans_first_in_line {
            result.push(current_word.clone());
            index += 1;
            continue;
        }

        let merged_text = format!(
            "{}{}",
            current_word.text.trim_end_matches('-'),
            next_word.text
        );
        let mut ghost_bbox = current_word.bounding_box;
        ghost_bbox.right += next_word.bounding_box.width();

        result.push(WordInfo {
            text: merged_text,
            confidence: current_word.confidence.min(next_word.confidence),
            bounding_box: current_word.bounding_box,
            ghost_bbox: Some(ghost_bbox),
            is_first_in_line: current_word.is_first_in_line,
            is_last_in_line: true,
            is_last_in_para: next_word.is_last_in_para,
        });

        if index + 2 >= words.len() {
            index += 2;
            continue;
        }

        let next_after_merged = &words[index + 2];
        let mut expanded_bbox = next_word.bounding_box;
        expanded_bbox.union(next_after_merged.bounding_box);
        result.push(WordInfo {
            bounding_box: expanded_bbox,
            is_first_in_line: true,
            ..next_after_merged.clone()
        });
        index += 3;
    }

    result
}

pub fn build_text_blocks(
    detected_words: &[DetectedWord],
    min_confidence: u32,
    join_without_spaces: bool,
    relax_single_char_confidence: bool,
) -> Vec<TextBlock> {
    let effective_min_confidence = if relax_single_char_confidence {
        (min_confidence.min(60)) as f32
    } else {
        min_confidence as f32
    };

    let all_words = detected_words
        .iter()
        .map(|word| WordInfo {
            text: word.text.clone(),
            confidence: word.confidence,
            bounding_box: word.bounding_box,
            ghost_bbox: None,
            is_first_in_line: word.is_at_beginning_of_para,
            is_last_in_line: word.end_line,
            is_last_in_para: word.end_para,
        })
        .collect::<Vec<_>>();

    let mut filtered_words = Vec::new();
    let mut pending_first_in_line = false;
    for (index, word) in all_words.iter().enumerate() {
        let should_include = word.confidence >= effective_min_confidence
            && (relax_single_char_confidence
                || !(word.text.chars().count() == 1
                    && word.confidence < (effective_min_confidence + 5.0).min(100.0)));

        if should_include {
            filtered_words.push(WordInfo {
                is_first_in_line: word.is_first_in_line || pending_first_in_line,
                ..word.clone()
            });
            pending_first_in_line = false;
        } else {
            if word.is_first_in_line {
                pending_first_in_line = true;
            }
            if word.is_last_in_line && index > 0 {
                if let Some(previous) = filtered_words.last_mut() {
                    previous.is_last_in_line = true;
                }
            }
            if word.is_last_in_para && index > 0 {
                if let Some(previous) = filtered_words.last_mut() {
                    previous.is_last_in_para = true;
                }
            }
        }
    }

    let filtered_words = merge_hyphenated_words(filtered_words);
    let mut blocks = Vec::new();
    let mut lines = Vec::new();
    let mut current_line: Option<TextLine> = None;
    let mut last_right = 0u32;

    for word in filtered_words {
        if word.text.trim().is_empty() {
            continue;
        }

        let real_bbox = word.ghost_bbox.unwrap_or(word.bounding_box);
        let skipped_first_word = current_line
            .as_ref()
            .is_some_and(|line| word.bounding_box.right < line.bounding_box.left);
        let first_word_in_line = word.is_first_in_line || skipped_first_word;
        let last_word_in_line = word.is_last_in_line;
        let last_word_in_para = word.is_last_in_para;

        if first_word_in_line || current_line.is_none() {
            current_line = Some(TextLine {
                text: word.text.clone(),
                bounding_box: word.bounding_box,
                oriented_box: OrientedRect::axis_aligned(word.bounding_box),
                tight_box: OrientedRect::axis_aligned(word.bounding_box),
                word_rects: vec![word.bounding_box],
            });
        } else if let Some(line) = current_line.as_mut() {
            let delta = word.bounding_box.left.saturating_sub(last_right);
            let char_width = real_bbox.width() as f32 / word.text.chars().count().max(1) as f32;
            let delta_in_chars = if char_width > 0.0 {
                delta as f32 / char_width
            } else {
                0.0
            };

            if delta_in_chars >= 3.0 {
                lines.push(line.clone());
                *line = TextLine {
                    text: word.text.clone(),
                    bounding_box: word.bounding_box,
                    oriented_box: OrientedRect::axis_aligned(word.bounding_box),
                    tight_box: OrientedRect::axis_aligned(word.bounding_box),
                    word_rects: vec![word.bounding_box],
                };
                if !lines.is_empty() {
                    blocks.push(TextBlock {
                        lines: std::mem::take(&mut lines),
                    });
                }
            } else {
                if join_without_spaces || line.text.is_empty() {
                    line.text.push_str(&word.text);
                } else {
                    line.text.push(' ');
                    line.text.push_str(&word.text);
                }
                line.word_rects.push(word.bounding_box);
                line.bounding_box.union(word.bounding_box);
                // Tesseract path: words are axis-aligned, so the line's oriented box is just
                // the AABB lifted. PPOCR detects whole lines as one "word" and never reaches
                // this merge branch.
                line.oriented_box = OrientedRect::axis_aligned(line.bounding_box);
                line.tight_box = OrientedRect::axis_aligned(line.bounding_box);
            }
        }

        if last_word_in_line {
            if let Some(line) = current_line.take() {
                if !line.text.trim().is_empty() {
                    lines.push(line);
                }
            }
        }

        if last_word_in_para && !lines.is_empty() {
            blocks.push(TextBlock {
                lines: std::mem::take(&mut lines),
            });
        }

        last_right = word.bounding_box.right;
    }

    if let Some(line) = current_line.take() {
        if !line.text.trim().is_empty() {
            lines.push(line);
        }
    }
    if !lines.is_empty() {
        blocks.push(TextBlock { lines });
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::{
        DetectedWord, OrientedRect, OverlayLayoutMode, ParagraphGroupingOptions, Rect, TextBlock,
        TextLine, build_text_blocks, group_lines_into_paragraphs, prepare_overlay_image,
    };
    use crate::{BackgroundMode, ReadingOrder};

    fn line(text: &str, left: u32, top: u32, right: u32, bottom: u32, tight_h: f32) -> TextLine {
        let rect = Rect {
            left,
            top,
            right,
            bottom,
        };
        let cx = (left + right) as f32 * 0.5;
        let cy = (top + bottom) as f32 * 0.5;
        let width = (right - left) as f32;
        let tight = OrientedRect {
            cx,
            cy,
            width,
            height: tight_h,
            angle_radians: 0.0,
        };
        TextLine {
            text: text.to_string(),
            bounding_box: rect,
            oriented_box: OrientedRect::axis_aligned(rect),
            tight_box: tight,
            word_rects: vec![rect],
        }
    }

    fn word(
        text: &str,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        is_first_in_line: bool,
        is_last_in_line: bool,
        is_last_in_para: bool,
    ) -> DetectedWord {
        DetectedWord {
            text: text.to_string(),
            confidence: 95.0,
            bounding_box: Rect {
                left,
                top,
                right,
                bottom,
            },
            is_at_beginning_of_para: is_first_in_line,
            end_line: is_last_in_line,
            end_para: is_last_in_para,
        }
    }

    #[test]
    fn translation_text_flattens_wrapped_lines_into_one_paragraph() {
        let detected_words = vec![
            word("relax", 52, 129, 103, 145, true, false, false),
            word("slightly", 115, 129, 192, 150, false, false, false),
            word("as", 202, 134, 224, 144, false, false, false),
            word("the", 235, 129, 267, 145, false, false, false),
            word("reprieve", 279, 133, 365, 150, false, false, false),
            word("of", 376, 128, 395, 144, false, false, false),
            word("warmth", 404, 127, 486, 144, false, false, false),
            word("began", 498, 127, 560, 148, false, false, false),
            word("to", 571, 129, 590, 143, false, false, false),
            word("press", 601, 131, 657, 148, false, false, false),
            word("against", 668, 128, 744, 147, false, true, false),
            word("his", 51, 158, 80, 174, true, false, false),
            word("frozen", 90, 159, 155, 174, false, false, false),
            word("cheeks.", 164, 158, 243, 174, false, true, true),
        ];

        let blocks = build_text_blocks(&detected_words, 75, false, false);
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0].source_text(),
            "relax slightly as the reprieve of warmth began to press against\nhis frozen cheeks."
        );
        assert_eq!(
            blocks[0].translation_text(),
            "relax slightly as the reprieve of warmth began to press against his frozen cheeks."
        );
    }

    #[test]
    fn prepare_overlay_image_erases_left_to_right_lines_without_touching_gap() {
        let width = 8;
        let height = 8;
        let gap_color = 0xFF12_34_56u32;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let color = if y == 3 { gap_color } else { 0xFF00_0000 };
            for _ in 0..width {
                rgba.extend_from_slice(&color.to_ne_bytes());
            }
        }

        let top_rect = Rect {
            left: 1,
            top: 1,
            right: 7,
            bottom: 3,
        };
        let bottom_rect = Rect {
            left: 1,
            top: 4,
            right: 7,
            bottom: 6,
        };
        let blocks = vec![TextBlock {
            lines: vec![
                TextLine {
                    text: "top".to_string(),
                    bounding_box: top_rect,
                    oriented_box: super::OrientedRect::axis_aligned(top_rect),
                    tight_box: super::OrientedRect::axis_aligned(top_rect),
                    word_rects: vec![top_rect],
                },
                TextLine {
                    text: "bottom".to_string(),
                    bounding_box: bottom_rect,
                    oriented_box: super::OrientedRect::axis_aligned(bottom_rect),
                    tight_box: super::OrientedRect::axis_aligned(bottom_rect),
                    word_rects: vec![bottom_rect],
                },
            ],
        }];
        let translated = vec!["translated text".to_string()];

        let prepared = prepare_overlay_image(
            &rgba,
            width,
            height,
            &blocks,
            &translated,
            BackgroundMode::BlackOnWhite,
            ReadingOrder::LeftToRight,
        )
        .expect("overlay should prepare");

        let gap_index = ((3 * width + 2) * 4) as usize;
        let gap_pixel = u32::from_ne_bytes(
            prepared.rgba_bytes[gap_index..gap_index + 4]
                .try_into()
                .expect("gap pixel"),
        );
        assert_eq!(gap_pixel, gap_color);

        let erased_index = ((1 * width + 2) * 4) as usize;
        let erased_pixel = u32::from_ne_bytes(
            prepared.rgba_bytes[erased_index..erased_index + 4]
                .try_into()
                .expect("erased pixel"),
        );
        assert_eq!(erased_pixel, 0xFFFF_FFFF);
        assert_eq!(prepared.blocks[0].lines.len(), 2);
        assert_eq!(prepared.blocks[0].lines[0].foreground_argb, 0xFF00_0000);
        assert_eq!(
            prepared.blocks[0].layout_hints.layout_mode,
            OverlayLayoutMode::PerLine
        );
        assert_eq!(prepared.blocks[0].layout_hints.suggested_font_size_px, 2.0);
    }

    #[test]
    fn grouping_single_paragraph_with_similar_heights_and_normal_gap() {
        // Three lines, x-height 10, baseline-gap 4 (line height ~14 ⇒ gap_ratio ~ 0.4).
        // Same column. Should collapse to one paragraph.
        let lines = vec![
            line("first body line", 20, 10, 200, 24, 10.0),
            line("second body line", 20, 28, 200, 42, 10.0),
            line("third body line", 20, 46, 180, 60, 10.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].lines.len(), 3);
    }

    #[test]
    fn grouping_breaks_on_blank_line() {
        // Big vertical gap between line 1 and line 2 — gap is ~3.5 × tight_h, well past 1.8.
        let lines = vec![
            line("paragraph one", 20, 10, 200, 24, 10.0),
            line("paragraph two", 20, 70, 200, 84, 10.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lines.len(), 1);
        assert_eq!(blocks[1].lines.len(), 1);
    }

    #[test]
    fn grouping_breaks_on_heading_size_change() {
        // The big-font line should not glue to the body line beneath it.
        let lines = vec![
            line("Big Heading", 20, 10, 220, 40, 26.0),
            line("body line one", 20, 48, 200, 62, 10.0),
            line("body line two", 20, 66, 200, 80, 10.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lines.len(), 1);
        assert_eq!(blocks[1].lines.len(), 2);
    }

    #[test]
    fn grouping_keeps_first_line_indent_in_paragraph() {
        // First line indented by ~3× tight_h; body lines flush at left=20.
        let lines = vec![
            line("Indented opener", 50, 10, 200, 24, 10.0),
            line("flush body line", 20, 28, 200, 42, 10.0),
            line("more flush body", 20, 46, 180, 60, 10.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].lines.len(), 3);
    }

    #[test]
    fn grouping_breaks_at_column_change() {
        // Reading-order sort interleaves two columns, alternating: A0, B0, A1, B1.
        // The grouper should split them by left-edge mismatch / large overlap.
        let lines = vec![
            line("col A line 1", 20, 10, 180, 24, 10.0),
            line("col B line 1", 220, 10, 380, 24, 10.0),
            line("col A line 2", 20, 28, 180, 42, 10.0),
            line("col B line 2", 220, 28, 380, 42, 10.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lines.len(), 2);
        assert_eq!(blocks[1].lines.len(), 2);
        assert_eq!(blocks[0].lines[0].text, "col A line 1");
        assert_eq!(blocks[0].lines[1].text, "col A line 2");
        assert_eq!(blocks[1].lines[0].text, "col B line 1");
        assert_eq!(blocks[1].lines[1].text, "col B line 2");
    }

    #[test]
    fn grouping_does_not_split_paragraph_just_because_last_line_is_short() {
        let lines = vec![
            line("long full-width body line", 20, 10, 280, 24, 10.0),
            line("also long body line", 20, 28, 270, 42, 10.0),
            line("short tail.", 20, 46, 90, 60, 10.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].lines.len(), 3);
    }

    #[test]
    fn grouping_separates_heading_from_body_when_size_is_only_modestly_larger() {
        // Mirrors the Stadsdorp poster case: a heading whose tight-height is ~22% over the
        // body. The pre-`opening_height_tolerance` version (0.25) merged this pair; the new
        // 0.15 opening gate must split them.
        let lines = vec![
            line("Section heading", 20, 10, 200, 34, 24.4),
            line("body line one", 20, 38, 200, 56, 20.0),
            line("body line two", 20, 60, 200, 78, 20.0),
        ];
        let blocks = group_lines_into_paragraphs(lines, ParagraphGroupingOptions::default());
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lines.len(), 1);
        assert_eq!(blocks[0].lines[0].text, "Section heading");
        assert_eq!(blocks[1].lines.len(), 2);
    }

    #[test]
    fn translation_text_drops_hyphen_for_broken_word() {
        let block = TextBlock {
            lines: vec![
                line("self-", 20, 10, 90, 24, 10.0),
                line("awareness builds", 20, 28, 200, 42, 10.0),
            ],
        };
        assert_eq!(block.translation_text(), "selfawareness builds");
    }

    #[test]
    fn translation_text_joins_cjk_without_space() {
        let block = TextBlock {
            lines: vec![
                line("今日は晴れ", 20, 10, 200, 28, 16.0),
                line("ています。", 20, 32, 200, 50, 16.0),
            ],
        };
        assert_eq!(block.translation_text(), "今日は晴れています。");
    }

    #[test]
    fn translation_text_joins_latin_with_space() {
        let block = TextBlock {
            lines: vec![
                line("hello world", 20, 10, 200, 24, 10.0),
                line("from rust", 20, 28, 200, 42, 10.0),
            ],
        };
        assert_eq!(block.translation_text(), "hello world from rust");
    }
}

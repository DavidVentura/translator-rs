#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Rect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextLine {
    pub text: String,
    pub bounding_box: Rect,
    pub word_rects: Vec<Rect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct PreparedTextLine {
    pub text: String,
    pub bounding_box: Rect,
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

impl TextBlock {
    pub fn source_text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn translation_text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.text.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
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
        let total = block
            .lines
            .iter()
            .map(|line| match reading_order {
                ReadingOrder::LeftToRight => line.bounding_box.height() as f32,
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
}

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

    let mid_x = (bounds.left + bounds.right) / 2;
    let mid_y = (bounds.top + bounds.bottom) / 2;
    for y in bounds.top..bounds.bottom {
        for x in bounds.left..bounds.right {
            let pixel = image.pixel_argb(x, y);
            let lum = luminance_u8(pixel);
            let r = channel_r(pixel) as u64;
            let g = channel_g(pixel) as u64;
            let b = channel_b(pixel) as u64;

            if lum.abs_diff(paper_peak) <= BG_BAND {
                let qi = (usize::from(x >= mid_x)) | (usize::from(y >= mid_y) << 1);
                let a = &mut bg_quad[qi];
                a.r += r;
                a.g += g;
                a.b += b;
                a.n += 1;
            }

            let is_core_fg = if bg_is_bright {
                lum <= fg_cutoff
            } else {
                lum >= fg_cutoff
            };
            if is_core_fg {
                fg_total.r += r;
                fg_total.g += g;
                fg_total.b += b;
                fg_total.n += 1;
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
    text_bounds: Rect,
    background_mode: crate::BackgroundMode,
) -> OverlayColors {
    match background_mode {
        crate::BackgroundMode::WhiteOnBlack => {
            let colors = OverlayColors {
                background_argb: argb(0, 0, 0),
                foreground_argb: argb(255, 255, 255),
            };
            image.fill_rect(text_bounds, colors.background_argb);
            colors
        }
        crate::BackgroundMode::BlackOnWhite => {
            let colors = OverlayColors {
                background_argb: argb(255, 255, 255),
                foreground_argb: argb(0, 0, 0),
            };
            image.fill_rect(text_bounds, colors.background_argb);
            colors
        }
        crate::BackgroundMode::AutoDetect => {
            let paint = autodetect_paint(&image.as_image(), text_bounds);
            image.apply_fill_plan(text_bounds, paint.fill);
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
                    let colors = erase_text_region(&mut image, line.bounding_box, background_mode);
                    if index == 0 {
                        block_background = colors.background_argb;
                        block_foreground = colors.foreground_argb;
                    }
                    prepared_lines.push(PreparedTextLine {
                        text: line.text.clone(),
                        bounding_box: line.bounding_box,
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
                let colors = erase_text_region(&mut image, block_bounds, background_mode);
                let prepared_lines = block
                    .lines
                    .iter()
                    .map(|line| PreparedTextLine {
                        text: line.text.clone(),
                        bounding_box: line.bounding_box,
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
        DetectedWord, OverlayLayoutMode, Rect, TextBlock, TextLine, build_text_blocks,
        prepare_overlay_image,
    };
    use crate::{BackgroundMode, ReadingOrder};

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

        let blocks = vec![TextBlock {
            lines: vec![
                TextLine {
                    text: "top".to_string(),
                    bounding_box: Rect {
                        left: 1,
                        top: 1,
                        right: 7,
                        bottom: 3,
                    },
                    word_rects: vec![Rect {
                        left: 1,
                        top: 1,
                        right: 7,
                        bottom: 3,
                    }],
                },
                TextLine {
                    text: "bottom".to_string(),
                    bounding_box: Rect {
                        left: 1,
                        top: 4,
                        right: 7,
                        bottom: 6,
                    },
                    word_rects: vec![Rect {
                        left: 1,
                        top: 4,
                        right: 7,
                        bottom: 6,
                    }],
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
}

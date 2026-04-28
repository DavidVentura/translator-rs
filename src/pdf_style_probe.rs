//! Pre-translation pass that walks each page's content stream and emits
//! per-`Tj` style samples (origin in user space + bold / italic / monospace
//! flags). Used by [`pdf_translate`] to enrich mupdf-extracted fragments
//! with intra-block style spans, so bold words in the middle of a paragraph
//! survive translation.
//!
//! [`pdf_translate`]: crate::pdf_translate

use std::collections::HashMap;

use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object, ObjectId};

#[derive(Debug)]
pub enum StyleProbeError {
    Lopdf(lopdf::Error),
}

impl std::fmt::Display for StyleProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lopdf(e) => write!(f, "lopdf: {e}"),
        }
    }
}

impl std::error::Error for StyleProbeError {}

impl From<lopdf::Error> for StyleProbeError {
    fn from(value: lopdf::Error) -> Self {
        Self::Lopdf(value)
    }
}

#[derive(Debug, Clone)]
pub struct TjSample {
    /// Glyph baseline-leading-edge in PDF user space.
    pub origin: (f32, f32),
    /// Per-axis scale of the combined `text_matrix × CTM` at this Tj. Multiply
    /// by the `Tf` font size to get the user-space rendered size.
    pub xy_scale: (f32, f32),
    /// `Tf` operand value (producer-local font size). The on-page size is
    /// `font_size × xy_scale.1`.
    pub font_size: f32,
    pub bold: bool,
    pub italic: bool,
    pub monospace: bool,
}

#[derive(Debug, Default, Clone)]
pub struct PageStyles {
    pub samples: Vec<TjSample>,
    /// Page `/Rotate` normalised to 0/90/180/270.
    pub rotate: i32,
    /// MediaBox dimensions in user space (independent of `/Rotate`).
    pub user_w: f32,
    pub user_h: f32,
}

impl PageStyles {
    /// Convert a sample's user-space origin to display coords (top-left
    /// origin), matching what mupdf's stext API reports for chars / lines.
    pub fn to_display(&self, user: (f32, f32)) -> (f32, f32) {
        match self.rotate {
            0 => (user.0, self.user_h - user.1),
            90 => (user.1, user.0),
            180 => (self.user_w - user.0, user.1),
            270 => (self.user_h - user.1, self.user_w - user.0),
            _ => (user.0, self.user_h - user.1),
        }
    }
}

/// Probe every page's content stream for per-Tj style info.
pub fn probe_pages(pdf_bytes: &[u8]) -> Result<Vec<PageStyles>, StyleProbeError> {
    let doc = Document::load_mem(pdf_bytes)?;
    let pages: Vec<ObjectId> = doc.get_pages().into_iter().map(|(_, id)| id).collect();
    pages.iter().map(|id| probe_page(&doc, *id)).collect()
}

fn probe_page(doc: &Document, page_id: ObjectId) -> Result<PageStyles, StyleProbeError> {
    let content = doc.get_and_decode_page_content(page_id)?;
    let fonts = doc.get_page_fonts(page_id).ok();
    let font_advances = FontAdvanceMap::from_page(doc, page_id);
    let (rotate, user_w, user_h) = read_page_geometry(doc, page_id);

    // Memoise font_resource -> (bold, italic, monospace) so we resolve each
    // font dict once per page.
    let mut flag_cache: HashMap<Vec<u8>, (bool, bool, bool)> = HashMap::new();
    let mut samples = Vec::new();
    walk_content(&content, &font_advances, |state, _op| {
        let flags = match &state.font_resource {
            Some(name) => {
                if let Some(cached) = flag_cache.get(name) {
                    *cached
                } else {
                    let resolved = fonts
                        .as_ref()
                        .and_then(|f| f.get(name.as_slice()))
                        .map(|d| font_flags(doc, d))
                        .unwrap_or((false, false, false));
                    flag_cache.insert(name.clone(), resolved);
                    resolved
                }
            }
            None => (false, false, false),
        };
        let combined = state.text_matrix.mul(state.ctm);
        let x_scale = (combined.a * combined.a + combined.b * combined.b).sqrt();
        let y_scale = (combined.c * combined.c + combined.d * combined.d).sqrt();
        let safe_x = if x_scale > 1e-6 { x_scale } else { 1.0 };
        let safe_y = if y_scale > 1e-6 { y_scale } else { 1.0 };
        samples.push(TjSample {
            origin: combined.transform_point(0.0, 0.0),
            xy_scale: (safe_x, safe_y),
            font_size: state.font_size,
            bold: flags.0,
            italic: flags.1,
            monospace: flags.2,
        });
    });
    Ok(PageStyles {
        samples,
        rotate,
        user_w,
        user_h,
    })
}

fn read_page_geometry(doc: &Document, page_id: ObjectId) -> (i32, f32, f32) {
    let page = doc.get_object(page_id).and_then(Object::as_dict);
    let rotate = page
        .as_ref()
        .ok()
        .and_then(|p| p.get(b"Rotate").ok())
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(0);
    let rotate = ((rotate % 360 + 360) % 360) as i32;
    let mut user_w = 612.0;
    let mut user_h = 792.0;
    if let Ok(p) = page
        && let Ok(Object::Array(arr)) = p.get(b"MediaBox")
        && arr.len() == 4
    {
        let nums: Option<Vec<f32>> = arr.iter().map(object_as_f32).collect();
        if let Some(n) = nums {
            user_w = n[2] - n[0];
            user_h = n[3] - n[1];
        }
    }
    (rotate, user_w, user_h)
}

#[derive(Debug, Clone)]
struct FontAdvance {
    code_bytes: usize,
    default_width: f32,
    widths: HashMap<u16, f32>,
}

impl Default for FontAdvance {
    fn default() -> Self {
        Self {
            code_bytes: 1,
            default_width: 500.0,
            widths: HashMap::new(),
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
        let mut widths = HashMap::new();
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
                widths: HashMap::new(),
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
            widths: HashMap::new(),
        };
        if let Ok(Object::Array(w_array)) = descendant.get(b"W") {
            parse_cid_widths(w_array, &mut out.widths);
        }
        out
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
                width += self
                    .widths
                    .get(&code)
                    .copied()
                    .unwrap_or(self.default_width);
                glyphs += 1;
                if code == 32 {
                    spaces += 1;
                }
            }
        } else {
            for &b in bytes {
                width += self
                    .widths
                    .get(&(b as u16))
                    .copied()
                    .unwrap_or(self.default_width);
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
    by_resource: HashMap<Vec<u8>, FontAdvance>,
}

impl FontAdvanceMap {
    fn from_page(doc: &Document, page_id: ObjectId) -> Self {
        let mut by_resource = HashMap::new();
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

fn parse_cid_widths(w_array: &[Object], widths: &mut HashMap<u16, f32>) {
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

// ---------------------------------------------------------------------------
// Minimal content-stream state machine (style-only, no graphics-state stack
// payload beyond CTM + font). filter_text_ops in pdf_write does the full
// version with colors and surgery; we only need positioning + font.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) struct Matrix {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Matrix {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }
    pub fn translate(tx: f32, ty: f32) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: tx,
            f: ty,
        }
    }
    pub fn mul(self, other: Matrix) -> Matrix {
        Matrix {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            e: self.e * other.a + self.f * other.c + other.e,
            f: self.e * other.b + self.f * other.d + other.f,
        }
    }
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }
}

#[derive(Debug, Clone)]
struct GraphicsState {
    ctm: Matrix,
    font_resource: Option<Vec<u8>>,
    font_size: f32,
    char_spacing: f32,
    word_spacing: f32,
    horizontal_scaling: f32,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            ctm: Matrix::identity(),
            font_resource: None,
            font_size: 12.0,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 1.0,
        }
    }
}

pub(crate) struct WalkState {
    stack: Vec<GraphicsState>,
    current: GraphicsState,
    pub text_matrix: Matrix,
    text_line_matrix: Matrix,
    text_leading: f32,
}

// Expose ctm + font_resource + font_size at sample time without leaking the
// whole struct.
pub(crate) struct SampleView<'a> {
    pub text_matrix: Matrix,
    pub ctm: Matrix,
    pub font_resource: &'a Option<Vec<u8>>,
    pub font_size: f32,
}

fn walk_content(
    content: &Content,
    font_advances: &FontAdvanceMap,
    mut on_show: impl FnMut(SampleView<'_>, &lopdf::content::Operation),
) {
    let mut state = WalkState {
        stack: Vec::new(),
        current: GraphicsState::default(),
        text_matrix: Matrix::identity(),
        text_line_matrix: Matrix::identity(),
        text_leading: 0.0,
    };
    for op in &content.operations {
        match op.operator.as_str() {
            "q" => state.stack.push(state.current.clone()),
            "Q" => {
                if let Some(prev) = state.stack.pop() {
                    state.current = prev;
                }
            }
            "cm" => {
                if let Some(m) = matrix_from_operands(&op.operands) {
                    state.current.ctm = m.mul(state.current.ctm);
                }
            }
            "BT" => {
                state.text_matrix = Matrix::identity();
                state.text_line_matrix = Matrix::identity();
            }
            "Tf" => {
                if let [Object::Name(name), size] = op.operands.as_slice() {
                    state.current.font_resource = Some(name.clone());
                    state.current.font_size =
                        object_as_f32(size).unwrap_or(state.current.font_size);
                }
            }
            "Tc" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.current.char_spacing = v;
                }
            }
            "Tw" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.current.word_spacing = v;
                }
            }
            "Tz" => {
                if let Some(v) = op.operands.first().and_then(object_as_f32) {
                    state.current.horizontal_scaling = v / 100.0;
                }
            }
            "Tm" => {
                if let Some(m) = matrix_from_operands(&op.operands) {
                    state.text_matrix = m;
                    state.text_line_matrix = m;
                }
            }
            "Td" | "TD" => {
                if op.operands.len() == 2 {
                    let tx = object_as_f32(&op.operands[0]).unwrap_or(0.0);
                    let ty = object_as_f32(&op.operands[1]).unwrap_or(0.0);
                    if op.operator == "TD" {
                        state.text_leading = -ty;
                    }
                    let new_lm = Matrix::translate(tx, ty).mul(state.text_line_matrix);
                    state.text_line_matrix = new_lm;
                    state.text_matrix = new_lm;
                }
            }
            "TL" => {
                if let Some(leading) = op.operands.first().and_then(object_as_f32) {
                    state.text_leading = leading;
                }
            }
            "T*" => {
                let leading = state.text_leading;
                let new_lm = Matrix::translate(0.0, -leading).mul(state.text_line_matrix);
                state.text_line_matrix = new_lm;
                state.text_matrix = new_lm;
            }
            "'" | "\"" => {
                if op.operator == "\"" {
                    if let Some(word_spacing) = op.operands.first().and_then(object_as_f32) {
                        state.current.word_spacing = word_spacing;
                    }
                    if let Some(char_spacing) = op.operands.get(1).and_then(object_as_f32) {
                        state.current.char_spacing = char_spacing;
                    }
                }
                let leading = state.text_leading;
                let new_lm = Matrix::translate(0.0, -leading).mul(state.text_line_matrix);
                state.text_line_matrix = new_lm;
                state.text_matrix = new_lm;
                on_show(
                    SampleView {
                        text_matrix: state.text_matrix,
                        ctm: state.current.ctm,
                        font_resource: &state.current.font_resource,
                        font_size: state.current.font_size,
                    },
                    op,
                );
                let advance = text_show_advance(op, &state, font_advances);
                state.text_matrix = Matrix::translate(advance, 0.0).mul(state.text_matrix);
                continue;
            }
            "Tj" | "TJ" => {
                on_show(
                    SampleView {
                        text_matrix: state.text_matrix,
                        ctm: state.current.ctm,
                        font_resource: &state.current.font_resource,
                        font_size: state.current.font_size,
                    },
                    op,
                );
                let advance = text_show_advance(op, &state, font_advances);
                state.text_matrix = Matrix::translate(advance, 0.0).mul(state.text_matrix);
            }
            _ => {}
        }
    }
}

fn text_show_advance(
    op: &lopdf::content::Operation,
    state: &WalkState,
    font_advances: &FontAdvanceMap,
) -> f32 {
    let font = font_advances.get(state.current.font_resource.as_ref());
    let mut text_advance = 0.0f32;
    let add_string = |bytes: &[u8]| {
        let (width_1000, glyphs, spaces) = font.string_width_1000(bytes);
        width_1000 * state.current.font_size / 1000.0
            + state.current.char_spacing * glyphs as f32
            + state.current.word_spacing * spaces as f32
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
                                text_advance -= adjustment * state.current.font_size / 1000.0;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    }

    text_advance * state.current.horizontal_scaling
}

fn matrix_from_operands(ops: &[Object]) -> Option<Matrix> {
    if ops.len() != 6 {
        return None;
    }
    Some(Matrix {
        a: object_as_f32(&ops[0])?,
        b: object_as_f32(&ops[1])?,
        c: object_as_f32(&ops[2])?,
        d: object_as_f32(&ops[3])?,
        e: object_as_f32(&ops[4])?,
        f: object_as_f32(&ops[5])?,
    })
}

fn object_as_f32(obj: &Object) -> Option<f32> {
    match obj {
        Object::Integer(i) => Some(*i as f32),
        Object::Real(r) => Some(*r),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Font flag resolution. Mirrors `pdf_write::font_flags` but stays inline so
// this module is independent.
// ---------------------------------------------------------------------------

/// `(bold, italic, monospace)` from the font's `/FontDescriptor /Flags`,
/// falling back to BaseFont-name pattern matching.
fn font_flags(doc: &Document, font_dict: &Dictionary) -> (bool, bool, bool) {
    let descriptor = font_dict.get(b"FontDescriptor").ok().and_then(|o| match o {
        Object::Reference(id) => doc.get_object(*id).ok().and_then(|o| o.as_dict().ok()),
        Object::Dictionary(d) => Some(d),
        _ => None,
    });

    if let Some(descriptor) = descriptor {
        let flags = descriptor
            .get(b"Flags")
            .ok()
            .and_then(|o| o.as_i64().ok())
            .unwrap_or(0);
        let monospace = (flags & (1 << 0)) != 0;
        let italic = (flags & (1 << 6)) != 0;
        let bold = (flags & (1 << 18)) != 0;
        if monospace || italic || bold {
            return (bold, italic, monospace);
        }
    }
    let base_font = font_dict.get(b"BaseFont").ok().and_then(|o| match o {
        Object::Name(name) => Some(name.as_slice()),
        _ => None,
    });
    base_font
        .map(detect_from_name)
        .unwrap_or((false, false, false))
}

fn detect_from_name(base_font: &[u8]) -> (bool, bool, bool) {
    let name = String::from_utf8_lossy(base_font);
    let lower = name.to_lowercase();
    let bold = ["bold", "heavy", "black", "semibold", "demibold"]
        .iter()
        .any(|k| lower.contains(k));
    let italic = ["italic", "oblique"].iter().any(|k| lower.contains(k));
    let monospace = [
        "courier",
        "mono",
        "consolas",
        "menlo",
        "jetbrainsmono",
        "sourcecodepro",
        "fixedsys",
    ]
    .iter()
    .any(|k| lower.contains(k));
    (bold, italic, monospace)
}

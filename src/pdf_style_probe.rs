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
    let (rotate, user_w, user_h) = read_page_geometry(doc, page_id);

    // Memoise font_resource -> (bold, italic, monospace) so we resolve each
    // font dict once per page.
    let mut flag_cache: HashMap<Vec<u8>, (bool, bool, bool)> = HashMap::new();
    let mut samples = Vec::new();
    walk_content(&content, |state, _op| {
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
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            ctm: Matrix::identity(),
            font_resource: None,
            font_size: 12.0,
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
                    if let Some(leading) = op.operands.first().and_then(object_as_f32) {
                        state.text_leading = leading;
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
            }
            _ => {}
        }
    }
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

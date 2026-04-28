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

/// PDF resource name we register Helvetica under.
const HELVETICA_RESOURCE_NAME: &[u8] = b"TrHelv";

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
pub fn write_translated_pdf(
    original_pdf_bytes: &[u8],
    translations: &[PageTranslationResult],
) -> Result<Vec<u8>, PdfWriteError> {
    let mut doc = Document::load_mem(original_pdf_bytes)?;

    let pages: Vec<(u32, ObjectId)> = doc.get_pages().into_iter().collect();

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
        // mupdf bboxes are in display coords (post-`/Rotate`, top-left
        // origin). Convert to PDF user space for the surgery pass.
        let removal_rects: Vec<UserRect> = translation
            .blocks
            .iter()
            .map(|b| user_rect_from_display(b.bounding_box, geom))
            .collect();

        let final_ctm = rewrite_page_content(&mut doc, *page_id, &removal_rects)?;
        ensure_helvetica_in_page_resources(&mut doc, *page_id)?;
        let overlay_stream =
            build_overlay_stream(&translation.blocks, &removal_rects, geom, final_ctm);
        append_content_stream(&mut doc, *page_id, overlay_stream)?;
    }

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
) -> Result<Matrix, PdfWriteError> {
    let content = doc.get_and_decode_page_content(page_id)?;
    let (filtered, final_ctm) = filter_text_ops(content.operations, removal_rects);
    let new_bytes = Content {
        operations: filtered,
    }
    .encode()?;
    doc.change_page_content(page_id, new_bytes)?;
    Ok(final_ctm)
}

/// Track text/graphics state across `ops` and drop any text-show op whose
/// current origin lies inside a removal rect. Returns the filtered op list
/// alongside the CTM still active after the (balanced) graphics-state stack
/// has unwound — i.e. what the appended content stream will inherit.
fn filter_text_ops(ops: Vec<Operation>, removal_rects: &[UserRect]) -> (Vec<Operation>, Matrix) {
    let mut state = State::new();
    let mut out = Vec::with_capacity(ops.len());

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
                if let Some(size) = op.operands.get(1).and_then(object_as_f32) {
                    state.font_size = size;
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
                    if let Some(leading) = op.operands.first().and_then(object_as_f32) {
                        state.text_leading = leading;
                    }
                    let leading = state.text_leading;
                    state.move_text(0.0, -leading);
                }

                let origin = state.current_text_origin();
                if removal_rects.iter().any(|r| r.contains(origin.0, origin.1)) {
                    // Drop the op; do not advance Tm by the string's width.
                    // Skipping that advance is safe because nothing downstream
                    // relies on the "post-show" cursor inside this BT/ET (PDF
                    // producers either explicitly set Tm/Td or end the BT block).
                    continue;
                }
            }
            _ => {}
        }
        out.push(op);
    }

    (out, state.current.ctm)
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

#[derive(Debug, Clone, Copy)]
struct GraphicsState {
    ctm: Matrix,
}

#[derive(Debug)]
struct State {
    stack: Vec<GraphicsState>,
    current: GraphicsState,
    in_text: bool,
    text_matrix: Matrix,
    text_line_matrix: Matrix,
    font_size: f32,
    text_leading: f32,
}

impl State {
    fn new() -> Self {
        Self {
            stack: Vec::new(),
            current: GraphicsState {
                ctm: Matrix::identity(),
            },
            in_text: false,
            text_matrix: Matrix::identity(),
            text_line_matrix: Matrix::identity(),
            font_size: 12.0,
            text_leading: 0.0,
        }
    }

    fn push(&mut self) {
        self.stack.push(self.current);
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

    fn current_text_origin(&self) -> (f32, f32) {
        // text origin = text_matrix × (0, 0) → then map through CTM.
        let (tx, ty) = self.text_matrix.transform_point(0.0, 0.0);
        self.current.ctm.transform_point(tx, ty)
    }
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

fn build_overlay_stream(
    blocks: &[TranslatedStyledBlock],
    user_rects: &[UserRect],
    geom: PageGeometry,
    final_ctm: Matrix,
) -> Vec<u8> {
    let mut out = Vec::<u8>::new();
    out.extend_from_slice(b"q\n");
    let inv_ctm = final_ctm.inverse().unwrap_or_else(Matrix::identity);
    for (block, user_rect) in blocks.iter().zip(user_rects.iter()) {
        emit_block(&mut out, block, *user_rect, geom, &inv_ctm);
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

    // Visual block dimensions: for /Rotate=±90 the user-space rect's x and y
    // swap their visual roles.
    let (vis_w, vis_h) = match geom.rotate {
        90 | 270 => (user_h, user_w),
        _ => (user_w, user_h),
    };

    let initial_font_size = (vis_h * (1.0 - TEXT_BASELINE_PAD)).max(4.0);
    let (font_size, lines) = wrap_to_fit(text, vis_w, vis_h, initial_font_size);
    let leading = font_size * LINE_HEIGHT_FACTOR;
    let total_height = leading * lines.len() as f32;
    let top_pad = ((vis_h - total_height).max(0.0)) * 0.5;
    // Distance from visual top of block to baseline of first line.
    let first_baseline_offset = top_pad + font_size;

    // Visual top-left corner expressed in user space, plus the user-space
    // direction vector for "next line down" (visually). Derived from the
    // forward user→display rotation in `user_rect_from_display`.
    let (visual_top_left_x, visual_top_left_y, line_dx, line_dy) = match geom.rotate {
        0 => (user_rect.x0, user_rect.y1, 0.0, -1.0),
        90 => (user_rect.x0, user_rect.y0, 1.0, 0.0),
        180 => (user_rect.x1, user_rect.y0, 0.0, 1.0),
        270 => (user_rect.x1, user_rect.y1, -1.0, 0.0),
        _ => (user_rect.x0, user_rect.y1, 0.0, -1.0),
    };

    let _ = writeln!(out, "0 0 0 rg");
    out.extend_from_slice(b"BT\n");
    let _ = writeln!(
        out,
        "/{} {:.2} Tf",
        std::str::from_utf8(HELVETICA_RESOURCE_NAME).unwrap(),
        font_size
    );

    for (i, line) in lines.iter().enumerate() {
        let off = first_baseline_offset + (i as f32) * leading;
        let user_x = visual_top_left_x + off * line_dx;
        let user_y = visual_top_left_y + off * line_dy;
        let (local_x, local_y) = inv_ctm.transform_point(user_x, user_y);
        let _ = writeln!(out, "1 0 0 1 {local_x:.2} {local_y:.2} Tm");
        out.extend_from_slice(b"(");
        write_pdf_string_body(out, line);
        out.extend_from_slice(b") Tj\n");
    }
    out.extend_from_slice(b"ET\n\n");
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

fn approx_text_width(text: &str, font_size: f32) -> f32 {
    text.chars().count() as f32 * font_size * HELVETICA_AVG_ADVANCE
}

fn wrap_to_fit(
    text: &str,
    max_width: f32,
    max_height: f32,
    mut font_size: f32,
) -> (f32, Vec<String>) {
    for _ in 0..6 {
        let lines = wrap_lines(text, max_width, font_size);
        let total_height = font_size * LINE_HEIGHT_FACTOR * lines.len() as f32;
        if total_height <= max_height || font_size <= 4.0 {
            return (font_size, lines);
        }
        font_size *= 0.85;
    }
    let final_size = font_size.max(4.0);
    (final_size, wrap_lines(text, max_width, final_size))
}

fn wrap_lines(text: &str, max_width: f32, font_size: f32) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        if approx_text_width(&candidate, font_size) <= max_width || current.is_empty() {
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

fn ensure_helvetica_in_page_resources(
    doc: &mut Document,
    page_id: ObjectId,
) -> Result<(), PdfWriteError> {
    let helv_id = doc.add_object({
        let mut d = Dictionary::new();
        d.set("Type", Object::Name(b"Font".to_vec()));
        d.set("Subtype", Object::Name(b"Type1".to_vec()));
        d.set("BaseFont", Object::Name(b"Helvetica".to_vec()));
        d.set("Encoding", Object::Name(b"WinAnsiEncoding".to_vec()));
        Object::Dictionary(d)
    });

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
    font_dict.set(HELVETICA_RESOURCE_NAME, Object::Reference(helv_id));
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

    let inline_resources = {
        let page = doc.get_object(page_id).and_then(Object::as_dict)?;
        match page.get(b"Resources") {
            Ok(Object::Dictionary(d)) => d.clone(),
            _ => Dictionary::new(),
        }
    };

    let new_id = doc.add_object(Object::Dictionary(inline_resources));
    let page_mut = doc.get_object_mut(page_id).and_then(Object::as_dict_mut)?;
    page_mut.set("Resources", Object::Reference(new_id));
    Ok(new_id)
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
        let lines = wrap_lines(text, 60.0, 10.0);
        assert!(lines.len() > 1);
        for line in &lines {
            if line.contains(' ') {
                let w = approx_text_width(line, 10.0);
                assert!(w <= 60.0, "line too wide: {line:?} width {w}");
            }
        }
    }

    #[test]
    fn empty_translations_does_not_touch_pdf() {
        let result = write_translated_pdf(b"", &[]);
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
        ensure_helvetica_in_page_resources(&mut doc, page_id).unwrap();

        let page = doc.get_object(page_id).unwrap().as_dict().unwrap();
        let resources_ref = page.get(b"Resources").unwrap();
        let resources_id = resources_ref.as_reference().unwrap();
        let resources = doc.get_object(resources_id).unwrap().as_dict().unwrap();
        let fonts = resources.get(b"Font").unwrap().as_dict().unwrap();
        let helv_ref = fonts.get(HELVETICA_RESOURCE_NAME).unwrap();
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
        let (filtered, _) = filter_text_ops(ops.clone(), &[rect]);
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
        let (filtered, _) = filter_text_ops(ops, &[rect]);
        assert!(filtered.iter().any(|o| o.operator == "Tj"));
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

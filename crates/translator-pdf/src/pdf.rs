//! Digital-PDF translation pipeline (extraction side).
//!
//! MuPDF provides the page geometry, text extraction, and optional debug
//! rendering used by the PDF smoke tests.

use crate::styled::TranslatedStyledBlock;
use mupdf::pdf::{PdfDocument, PdfWriteOptions};
use mupdf::{Buffer, Colorspace, Document, Error as MupdfError, Matrix};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageDims {
    pub width_pts: f32,
    pub height_pts: f32,
}

/// Maps coordinates between a letterboxed debug bitmap and the PDF page in
/// points.
///
/// Image axes: x→right, y→down, origin top-left.
#[derive(Debug, Clone, Copy)]
pub struct PageTransform {
    pub page: PageDims,
    pub bitmap_size: u32,
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl PageTransform {
    pub fn new(page: PageDims, bitmap_size: u32) -> Self {
        let target = bitmap_size as f32;
        let scale = (target / page.width_pts).min(target / page.height_pts);
        let scaled_w = page.width_pts * scale;
        let scaled_h = page.height_pts * scale;
        Self {
            page,
            bitmap_size,
            scale,
            pad_x: (target - scaled_w) * 0.5,
            pad_y: (target - scaled_h) * 0.5,
        }
    }
}

#[derive(Debug)]
pub struct RenderedPage {
    /// RGBA8, `bitmap_size` × `bitmap_size`, letterboxed to white.
    pub rgba: Vec<u8>,
    pub transform: PageTransform,
}

#[derive(Debug)]
pub enum PdfError {
    Mupdf(MupdfError),
    UnexpectedPixmap(String),
}

impl From<MupdfError> for PdfError {
    fn from(value: MupdfError) -> Self {
        Self::Mupdf(value)
    }
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mupdf(err) => write!(f, "mupdf: {err}"),
            Self::UnexpectedPixmap(msg) => write!(f, "unexpected pixmap: {msg}"),
        }
    }
}

impl std::error::Error for PdfError {}

/// The input PDF as re-serialized by MuPDF. MuPDF rebuilds a broken
/// cross-reference table while lopdf rejects the whole file, so every pass
/// reads this rewrite instead of the original bytes.
pub struct NormalizedPdf {
    buffer: Buffer,
}

impl NormalizedPdf {
    pub fn new(pdf_bytes: Vec<u8>) -> Result<Self, PdfError> {
        let document = PdfDocument::from_bytes(&pdf_bytes)?;
        // MuPDF keeps its own copy of the input, so releasing ours before
        // writing holds at most two copies of a large book in memory.
        drop(pdf_bytes);
        let buffer = document.write_with_options(PdfWriteOptions::default())?;
        Ok(Self { buffer })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

/// Renders every page of `pdf_bytes` into a square `bitmap_size × bitmap_size`
/// RGBA letterboxed bitmap for diagnostics and smoke-test dumps.
pub fn render_pages_for_debug(
    pdf_bytes: &[u8],
    bitmap_size: u32,
) -> Result<Vec<RenderedPage>, PdfError> {
    let document = Document::from_bytes(pdf_bytes, "application/pdf")?;
    let page_count = document.page_count()?;
    let mut pages = Vec::with_capacity(page_count as usize);
    for i in 0..page_count {
        let page = document.load_page(i)?;
        pages.push(render_page_for_debug(&page, bitmap_size)?);
    }
    Ok(pages)
}

fn render_page_for_debug(page: &mupdf::Page, bitmap_size: u32) -> Result<RenderedPage, PdfError> {
    let bounds = page.bounds()?;
    let dims = PageDims {
        width_pts: bounds.x1 - bounds.x0,
        height_pts: bounds.y1 - bounds.y0,
    };
    let transform = PageTransform::new(dims, bitmap_size);

    let ctm = Matrix::new_scale(transform.scale, transform.scale);
    let pixmap = page.to_pixmap(&ctm, &Colorspace::device_rgb(), true, false)?;

    if pixmap.n() != 4 {
        return Err(PdfError::UnexpectedPixmap(format!(
            "expected RGBA (n=4), got n={}",
            pixmap.n()
        )));
    }

    let rgba = letterbox_rgba(
        pixmap.samples(),
        pixmap.width(),
        pixmap.height(),
        bitmap_size,
        transform.pad_x.round() as u32,
        transform.pad_y.round() as u32,
    );

    Ok(RenderedPage { rgba, transform })
}

/// Pastes `src` (RGBA, `src_w × src_h`) into a fresh `target_size × target_size`
/// white RGBA canvas at offset `(pad_x, pad_y)`.
fn letterbox_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    target_size: u32,
    pad_x: u32,
    pad_y: u32,
) -> Vec<u8> {
    let target = target_size as usize;
    let mut canvas = vec![0xFFu8; target * target * 4];
    let src_w = src_w as usize;
    let src_h = src_h as usize;
    let pad_x = pad_x as usize;
    let pad_y = pad_y as usize;
    for row in 0..src_h {
        let src_offset = row * src_w * 4;
        let dst_offset = ((pad_y + row) * target + pad_x) * 4;
        let len = src_w * 4;
        canvas[dst_offset..dst_offset + len].copy_from_slice(&src[src_offset..src_offset + len]);
    }
    canvas
}

#[derive(Debug, Clone)]
pub struct PageTranslationResult {
    pub page_index: usize,
    pub page: PageDims,
    pub blocks: Vec<TranslatedStyledBlock>,
    pub error: Option<String>,
    /// The language the blocks were translated **into**, carrying the script
    /// the catalog records for it. The PDF writer hands the script straight to
    /// its [`FontProvider`] instead of inferring one from the tag.
    pub target_language: translator_core::api::ScriptedLanguage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_portrait_pads_horizontally() {
        let t = PageTransform::new(
            PageDims {
                width_pts: 612.0,
                height_pts: 792.0,
            },
            1024,
        );

        // Page is taller than wide → fits to height; pad_x > 0, pad_y == 0.
        assert!(t.pad_y.abs() < 0.5);
        assert!(t.pad_x > 0.0);
    }

    #[test]
    fn landscape_letterbox_pads_vertically() {
        let t = PageTransform::new(
            PageDims {
                width_pts: 1000.0,
                height_pts: 500.0,
            },
            1024,
        );
        assert!(t.pad_x.abs() < 0.5);
        assert!(t.pad_y > 0.0);
    }

    fn one_page_pdf(text: &str) -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{Object, Stream, dictionary};

        let mut doc = lopdf::Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 24.into()]),
                Operation::new("Td", vec![72.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal(text)]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font_id } },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    fn break_startxref(pdf: &[u8]) -> Vec<u8> {
        let at = pdf
            .windows(b"startxref".len())
            .rposition(|window| window == b"startxref")
            .unwrap();
        let mut broken = pdf[..at].to_vec();
        broken.extend_from_slice(b"startxref\n17\n%%EOF\n");
        broken
    }

    #[test]
    fn normalized_pdf_repairs_a_cross_reference_table_lopdf_rejects() {
        let broken = break_startxref(&one_page_pdf("Hello world"));
        assert!(lopdf::Document::load_mem(&broken).is_err());

        let normalized = NormalizedPdf::new(broken).unwrap();

        let reloaded = lopdf::Document::load_mem(normalized.as_bytes()).unwrap();
        assert_eq!(reloaded.get_pages().len(), 1);
        crate::pdf_write::write_translated_pdf(
            normalized.as_bytes(),
            &[],
            &translator_render::font_provider::NoFontProvider,
        )
        .unwrap();
    }
}

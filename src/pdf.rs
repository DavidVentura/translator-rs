//! Digital-PDF translation pipeline (extraction side).
//!
//! Renders pages to bitmaps for the DocLayout-YOLO ONNX model and exposes the
//! geometric transform needed to map detection bboxes back to PDF user-space
//! points.
//!
//! mupdf is the rasterizer; lopdf will own the editing/save side later.

use mupdf::{Colorspace, Document, Error as MupdfError, Matrix};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageDims {
    pub width_pts: f32,
    pub height_pts: f32,
}

/// Maps coordinates between the letterboxed model-input bitmap and the PDF
/// page in user-space points.
///
/// Image axes: x→right, y→down, origin top-left.
/// PDF axes:   x→right, y→up,   origin bottom-left.
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

    /// PDF user-space rect (points, bottom-left origin) → image-space bbox
    /// `(x0, y0, x1, y1)` in pixels (top-left origin), with `x0 ≤ x1, y0 ≤ y1`.
    pub fn pdf_bbox_to_image(&self, rect: &PdfRectPts) -> (f32, f32, f32, f32) {
        let x0 = rect.left * self.scale + self.pad_x;
        let x1 = rect.right * self.scale + self.pad_x;
        // PDF top is high y in points; image top is low y in pixels.
        let img_y_top = (self.page.height_pts - rect.top) * self.scale + self.pad_y;
        let img_y_bot = (self.page.height_pts - rect.bottom) * self.scale + self.pad_y;
        (
            x0.min(x1),
            img_y_top.min(img_y_bot),
            x0.max(x1),
            img_y_top.max(img_y_bot),
        )
    }

    /// Image-space bbox (pixels, top-left origin) → PDF user-space rect (points, bottom-left origin).
    pub fn image_bbox_to_pdf(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> PdfRectPts {
        let pdf_x0 = (x0 - self.pad_x) / self.scale;
        let pdf_x1 = (x1 - self.pad_x) / self.scale;
        let pdf_top_y = (y0 - self.pad_y) / self.scale;
        let pdf_bot_y = (y1 - self.pad_y) / self.scale;
        PdfRectPts {
            left: pdf_x0.clamp(0.0, self.page.width_pts),
            right: pdf_x1.clamp(0.0, self.page.width_pts),
            top: (self.page.height_pts - pdf_top_y).clamp(0.0, self.page.height_pts),
            bottom: (self.page.height_pts - pdf_bot_y).clamp(0.0, self.page.height_pts),
        }
    }
}

/// PDF user-space rectangle (points, bottom-left origin), with `top > bottom`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdfRectPts {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
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

/// Renders every page of `pdf_bytes` into a square `bitmap_size × bitmap_size`
/// RGBA letterboxed bitmap suitable as DocLayout-YOLO input.
pub fn render_pages_for_layout(
    pdf_bytes: &[u8],
    bitmap_size: u32,
) -> Result<Vec<RenderedPage>, PdfError> {
    let document = Document::from_bytes(pdf_bytes, "application/pdf")?;
    let page_count = document.page_count()?;
    let mut pages = Vec::with_capacity(page_count as usize);
    for i in 0..page_count {
        let page = document.load_page(i)?;
        pages.push(render_page_for_layout(&page, bitmap_size)?);
    }
    Ok(pages)
}

fn render_page_for_layout(page: &mupdf::Page, bitmap_size: u32) -> Result<RenderedPage, PdfError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_round_trip_corners() {
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

        // top-left of page in image → (pad_x, 0)
        let r = t.image_bbox_to_pdf(t.pad_x, 0.0, t.pad_x + 1.0, 1.0);
        assert!((r.left - 0.0).abs() < 0.01);
        assert!((r.top - 792.0).abs() < 0.01);

        // bottom-right of page in image → (pad_x + scaled_w, 1024)
        let scaled_w = 612.0 * t.scale;
        let r = t.image_bbox_to_pdf(t.pad_x + scaled_w - 1.0, 1023.0, t.pad_x + scaled_w, 1024.0);
        assert!((r.right - 612.0).abs() < 0.05);
        assert!((r.bottom - 0.0).abs() < 0.05);
    }

    #[test]
    fn pdf_to_image_round_trips_image_to_pdf() {
        let t = PageTransform::new(
            PageDims {
                width_pts: 612.0,
                height_pts: 792.0,
            },
            1024,
        );
        // Coordinates strictly inside the rendered page region (avoiding the
        // letterbox padding so the round-trip isn't clamped at the edge).
        let pad_x = t.pad_x.ceil();
        let scaled_w = 612.0 * t.scale;
        let img = (pad_x + 50.0, 50.0, pad_x + scaled_w - 50.0, 600.0);
        let pdf_rect = t.image_bbox_to_pdf(img.0, img.1, img.2, img.3);
        let back = t.pdf_bbox_to_image(&pdf_rect);
        assert!((back.0 - img.0).abs() < 0.05, "x0 {} vs {}", back.0, img.0);
        assert!((back.1 - img.1).abs() < 0.05, "y0 {} vs {}", back.1, img.1);
        assert!((back.2 - img.2).abs() < 0.05, "x1 {} vs {}", back.2, img.2);
        assert!((back.3 - img.3).abs() < 0.05, "y1 {} vs {}", back.3, img.3);
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
}

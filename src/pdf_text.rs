//! Extract text + bounding boxes from a PDF via mupdf's stext API.
//!
//! Each page yields a `Vec<StyledFragment>` ready to feed the existing
//! geometric clustering in `crate::styled`. v1 does **not** populate
//! style (color/bold/italic) — mupdf-rs's safe API doesn't surface those
//! fields. We can revisit when visual fidelity demands it.

use mupdf::text_page::TextBlockType;
use mupdf::{Document, Page, TextPageFlags};

use crate::ocr::Rect;
use crate::pdf::{PageDims, PdfError};
use crate::styled::StyledFragment;

#[derive(Debug, Clone)]
pub struct PageTextFragments {
    pub page_index: usize,
    pub page: PageDims,
    pub fragments: Vec<StyledFragment>,
}

const STEXT_FLAGS: TextPageFlags = TextPageFlags::from_bits_truncate(
    TextPageFlags::PRESERVE_LIGATURES.bits()
        | TextPageFlags::PRESERVE_WHITESPACE.bits()
        | TextPageFlags::DEHYPHENATE.bits()
        | TextPageFlags::ACCURATE_BBOXES.bits(),
);

/// One fragment per stext line. Fragments inside the same stext block share
/// `translation_group` so they translate as one unit.
pub fn extract_text(pdf_bytes: &[u8]) -> Result<Vec<PageTextFragments>, PdfError> {
    let document = Document::from_bytes(pdf_bytes, "application/pdf")?;
    let page_count = document.page_count()?;
    let mut pages = Vec::with_capacity(page_count as usize);
    for i in 0..page_count {
        let page = document.load_page(i)?;
        pages.push(extract_page(&page, i as usize)?);
    }
    Ok(pages)
}

fn extract_page(page: &Page, page_index: usize) -> Result<PageTextFragments, PdfError> {
    let bounds = page.bounds()?;
    let dims = PageDims {
        width_pts: bounds.x1 - bounds.x0,
        height_pts: bounds.y1 - bounds.y0,
    };

    let stext = page.to_text_page(STEXT_FLAGS)?;
    let mut fragments = Vec::new();

    for (block_index, block) in stext
        .blocks()
        .filter(|b| matches!(b.r#type(), TextBlockType::Text))
        .enumerate()
    {
        let translation_group = block_index as u32;
        for line in block.lines() {
            let mut text = String::new();
            for ch in line.chars() {
                if let Some(c) = ch.char() {
                    text.push(c);
                }
            }
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }

            let line_bbox = line.bounds();
            fragments.push(StyledFragment {
                text: trimmed.to_string(),
                bounding_box: rect_from_mupdf(line_bbox),
                style: None,
                layout_group: 0,
                translation_group,
                cluster_group: translation_group,
            });
        }
    }

    Ok(PageTextFragments {
        page_index,
        page: dims,
        fragments,
    })
}

/// mupdf::Rect (top-left origin, points, f32) → our Rect (top-left origin, u32).
/// PDFs use ≤842pt for typical sizes — integer points are fine for clustering.
/// Rounds outward (floor min, ceil max) so a fractional bbox doesn't shrink
/// past the actual text extent.
fn rect_from_mupdf(r: mupdf::Rect) -> Rect {
    Rect {
        left: r.x0.max(0.0).floor() as u32,
        top: r.y0.max(0.0).floor() as u32,
        right: r.x1.max(0.0).ceil() as u32,
        bottom: r.y1.max(0.0).ceil() as u32,
    }
}

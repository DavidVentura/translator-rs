//! Pre-translation pass that walks each page's content stream and emits
//! per-`Tj` style samples (origin in user space + bold / italic / monospace
//! flags). Used by [`pdf_translate`] to enrich mupdf-extracted fragments
//! with intra-block style spans, so bold words in the middle of a paragraph
//! survive translation.
//!
//! [`pdf_translate`]: crate::pdf_translate

use std::collections::HashMap;

use lopdf::{Document, ObjectId};

use crate::pdf_content::{
    ContentState, FontAdvanceMap, FontStyleFlags, PageGeometry, font_flags, is_text_show_operator,
};

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
    pub(crate) flags: FontStyleFlags,
}

#[derive(Debug, Clone)]
pub struct PageStyles {
    pub samples: Vec<TjSample>,
    pub(crate) geom: PageGeometry,
}

impl Default for PageStyles {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            geom: PageGeometry {
                user_w: 612.0,
                user_h: 792.0,
                rotate: 0,
            },
        }
    }
}

impl PageStyles {
    /// Convert a sample's user-space origin to display coords (top-left
    /// origin), matching what mupdf's stext API reports for chars / lines.
    pub fn to_display(&self, user: (f32, f32)) -> (f32, f32) {
        self.geom.to_display(user)
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
    let geom = PageGeometry::read(doc, page_id, None);

    // Memoise font_resource -> style flags so we resolve each font dict once
    // per page.
    let mut flag_cache: HashMap<Vec<u8>, FontStyleFlags> = HashMap::new();
    let mut samples = Vec::new();
    let mut state = ContentState::new();

    for op in &content.operations {
        if !is_text_show_operator(&op.operator) {
            state.apply_non_show_op(op);
            continue;
        }
        let snapshot = state.process_text_show(op, &font_advances);

        let flags = match state.font_resource() {
            Some(name) => {
                if let Some(cached) = flag_cache.get(name) {
                    *cached
                } else {
                    let resolved = fonts
                        .as_ref()
                        .and_then(|f| f.get(name.as_slice()))
                        .map(|d| font_flags(doc, d))
                        .unwrap_or_default();
                    flag_cache.insert(name.clone(), resolved);
                    resolved
                }
            }
            None => FontStyleFlags::default(),
        };

        let combined = snapshot.combined;
        let x_scale = (combined.a * combined.a + combined.b * combined.b).sqrt();
        let y_scale = (combined.c * combined.c + combined.d * combined.d).sqrt();
        let safe_x = if x_scale > 1e-6 { x_scale } else { 1.0 };
        let safe_y = if y_scale > 1e-6 { y_scale } else { 1.0 };
        samples.push(TjSample {
            origin: combined.transform_point(0.0, 0.0),
            xy_scale: (safe_x, safe_y),
            font_size: state.font_size(),
            flags,
        });
    }

    Ok(PageStyles { samples, geom })
}

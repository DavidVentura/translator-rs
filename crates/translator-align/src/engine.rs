use std::sync::Arc;

use translator_core::api::{TranslatorError, TranslatorErrorKind};
use translator_core::catalog::CatalogSnapshot;

use crate::doc_align::{DocAligner, DocumentDetection};
use crate::doc_align_refine::{QuadQuality, refine_quad_with_quality};

pub struct DocAlignEngine {
    warm: Option<(String, Arc<DocAligner>)>,
}

impl Default for DocAlignEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DocAlignEngine {
    pub fn new() -> Self {
        Self { warm: None }
    }

    fn aligner(&mut self, snap: &CatalogSnapshot) -> Result<Arc<DocAligner>, TranslatorError> {
        let files = snap.catalog.support_files_by_kind("doc_detect");
        let file = files.first().ok_or_else(|| {
            TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                "no doc_detect support pack in catalog",
            )
        })?;
        let abs_path = std::path::Path::new(&snap.base_dir).join(&file.install_path);
        let abs_str = abs_path.to_string_lossy().into_owned();
        if !abs_path.exists() {
            return Err(TranslatorError::new(
                TranslatorErrorKind::MissingAsset,
                format!("doc-align model not installed at {}", abs_str),
            ));
        }
        if let Some((path, aligner)) = self.warm.as_ref() {
            if path == &abs_str {
                return Ok(Arc::clone(aligner));
            }
        }
        let aligner = Arc::new(DocAligner::load(&abs_path, 2)?);
        self.warm = Some((abs_str, Arc::clone(&aligner)));
        Ok(aligner)
    }

    pub fn detect_document_quad(
        &mut self,
        snap: &CatalogSnapshot,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<DocumentDetection>, TranslatorError> {
        let Some(mut detection) = self.aligner(snap)?.detect(rgba, width, height)? else {
            return Ok(None);
        };
        let refined = refine_quad_with_quality(rgba, width, height, &detection.quad);
        if refined.quality == QuadQuality::Bad {
            return Ok(None);
        }
        detection.quad = refined.quad;
        Ok(Some(detection))
    }
}

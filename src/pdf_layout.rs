//! DocLayout-YOLO ONNX inference: classify each page region as
//! Title / Plain text / Figure / Table / Formula / etc.
//!
//! The model is a YOLOv10 fine-tune. It expects a square RGB tensor
//! `(1, 3, H, W)` with values in `[0, 1]`, and emits `(1, N, 6)` where each row
//! is `[x1, y1, x2, y2, score, class_id]` in input-pixel coordinates.
//! Built-in NMS — no post-processing required beyond a score threshold.

use std::path::Path;

use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

use crate::pdf::{PageTransform, PdfRectPts};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Xnnpack,
}

/// Build an ORT [`Session`] for the DocLayout-YOLO model.
///
/// Mirrors the piper-rs configuration: low intra/inter-op threads, spinning
/// disabled, optional XNNPACK execution provider with CPU fallback.
pub fn build_session(
    model_path: impl AsRef<Path>,
    backend: Backend,
) -> Result<Session, LayoutError> {
    let path = model_path.as_ref();
    let make = || Session::builder().map_err(|e| LayoutError::Ort(format!("session builder: {e}")));

    let builder = match backend {
        Backend::Cpu => make()?,
        Backend::Xnnpack => {
            match make()?.with_execution_providers([ort::ep::XNNPACK::default().build()]) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("ORT: XNNPACK not available ({e}), falling back to CPU");
                    make()?
                }
            }
        }
    };

    builder
        .with_intra_threads(2)
        .map_err(|e| LayoutError::Ort(format!("intra threads: {e}")))?
        .with_inter_threads(2)
        .map_err(|e| LayoutError::Ort(format!("inter threads: {e}")))?
        .with_intra_op_spinning(false)
        .map_err(|e| LayoutError::Ort(format!("disable intra spin: {e}")))?
        .with_inter_op_spinning(false)
        .map_err(|e| LayoutError::Ort(format!("disable inter spin: {e}")))?
        .commit_from_file(path)
        .map_err(|e| LayoutError::Ort(format!("load model {}: {e}", path.display())))
}

/// DocStructBench taxonomy (10 classes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutClass {
    Title,
    PlainText,
    Abandon,
    Figure,
    FigureCaption,
    Table,
    TableCaption,
    TableFootnote,
    IsolateFormula,
    FormulaCaption,
}

impl LayoutClass {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::Title,
            1 => Self::PlainText,
            2 => Self::Abandon,
            3 => Self::Figure,
            4 => Self::FigureCaption,
            5 => Self::Table,
            6 => Self::TableCaption,
            7 => Self::TableFootnote,
            8 => Self::IsolateFormula,
            9 => Self::FormulaCaption,
            _ => return None,
        })
    }

    /// Whether text inside this region should be translated.
    pub fn is_translatable_text(self) -> bool {
        matches!(
            self,
            Self::Title
                | Self::PlainText
                | Self::FigureCaption
                | Self::TableCaption
                | Self::TableFootnote
                | Self::FormulaCaption
        )
    }

    /// Whether text inside this region should be left alone (formulas, figures).
    pub fn is_skip(self) -> bool {
        matches!(self, Self::Figure | Self::IsolateFormula | Self::Abandon)
    }
}

#[derive(Debug, Clone)]
pub struct LayoutRegion {
    pub class: LayoutClass,
    pub score: f32,
    pub bbox: PdfRectPts,
}

#[derive(Debug)]
pub enum LayoutError {
    Ort(String),
    UnexpectedShape(String),
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ort(msg) => write!(f, "ort: {msg}"),
            Self::UnexpectedShape(msg) => write!(f, "unexpected output shape: {msg}"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Runs DocLayout-YOLO inference on a single letterboxed RGBA page bitmap.
///
/// `score_threshold` should typically be ~0.25.
pub fn detect_regions(
    session: &mut Session,
    rgba: &[u8],
    transform: &PageTransform,
    score_threshold: f32,
) -> Result<Vec<LayoutRegion>, LayoutError> {
    let size = transform.bitmap_size as usize;
    assert_eq!(
        rgba.len(),
        size * size * 4,
        "rgba buffer doesn't match transform.bitmap_size"
    );

    let input = rgba_to_chw_f32(rgba, size);
    let input_tensor = Tensor::<f32>::from_array((
        [1, 3, size, size],
        input.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .map_err(|e| LayoutError::Ort(format!("input tensor: {e}")))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| LayoutError::Ort(format!("run: {e}")))?;

    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| LayoutError::Ort(format!("extract: {e}")))?;

    decode_yolov10(shape, data, transform, score_threshold)
}

fn rgba_to_chw_f32(rgba: &[u8], size: usize) -> Array4<f32> {
    let mut chw = Array4::<f32>::zeros((1, 3, size, size));
    for y in 0..size {
        for x in 0..size {
            let i = (y * size + x) * 4;
            // YOLOv10 stock is RGB, /255, no mean/std.
            chw[[0, 0, y, x]] = rgba[i] as f32 / 255.0;
            chw[[0, 1, y, x]] = rgba[i + 1] as f32 / 255.0;
            chw[[0, 2, y, x]] = rgba[i + 2] as f32 / 255.0;
        }
    }
    chw
}

/// YOLOv10 output: `(1, N, 6)` with rows `[x1, y1, x2, y2, score, class]`.
fn decode_yolov10(
    shape: &[i64],
    data: &[f32],
    transform: &PageTransform,
    score_threshold: f32,
) -> Result<Vec<LayoutRegion>, LayoutError> {
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 6 {
        return Err(LayoutError::UnexpectedShape(format!("{shape:?}")));
    }

    let n = shape[1] as usize;
    let mut regions = Vec::new();
    for i in 0..n {
        let row = &data[i * 6..(i + 1) * 6];
        let score = row[4];
        if score < score_threshold {
            continue;
        }
        let Some(class) = LayoutClass::from_id(row[5] as u32) else {
            continue;
        };
        let bbox = transform.image_bbox_to_pdf(row[0], row[1], row[2], row[3]);
        regions.push(LayoutRegion { class, score, bbox });
    }

    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdf::PageDims;

    #[test]
    fn decode_filters_low_scores() {
        let transform = PageTransform::new(
            PageDims {
                width_pts: 612.0,
                height_pts: 792.0,
            },
            1024,
        );
        // Two rows: one above threshold, one below.
        let shape = [1, 2, 6];
        let data = [
            // x1, y1, x2, y2, score, class
            100.0, 100.0, 500.0, 200.0, 0.9, 1.0, // PlainText, kept
            100.0, 300.0, 500.0, 400.0, 0.1, 0.0, // Title, dropped
        ];
        let out = decode_yolov10(&shape, &data, &transform, 0.25).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class, LayoutClass::PlainText);
    }

    #[test]
    fn decode_unknown_class_skipped() {
        let transform = PageTransform::new(
            PageDims {
                width_pts: 612.0,
                height_pts: 792.0,
            },
            1024,
        );
        let shape = [1, 1, 6];
        let data = [0.0, 0.0, 10.0, 10.0, 0.99, 99.0];
        let out = decode_yolov10(&shape, &data, &transform, 0.25).unwrap();
        assert!(out.is_empty());
    }
}

use std::path::Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::api::{TranslatorError, TranslatorErrorKind};

pub(crate) fn load_onnx_session(
    model_path: &Path,
    intra_threads: usize,
) -> Result<Session, TranslatorError> {
    Session::builder()
        .and_then(|builder| builder.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|builder| builder.with_intra_threads(intra_threads))
        .and_then(|builder| builder.commit_from_file(model_path))
        .map_err(|error| {
            TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!(
                    "failed to load ONNX model at {}: {error}",
                    model_path.display()
                ),
            )
        })
}

use std::path::Path;

use mnn_sys::{InferenceConfig, InferenceEngine, MemoryMode, PrecisionMode};

use crate::api::{TranslatorError, TranslatorErrorKind};

pub(crate) struct MnnSession {
    engine: InferenceEngine,
}

impl MnnSession {
    pub fn load(model_path: &Path, intra_threads: usize) -> Result<Self, TranslatorError> {
        Self::load_with_modes(
            model_path,
            intra_threads,
            PrecisionMode::Low,
            MemoryMode::Normal,
        )
    }

    pub fn load_with_modes(
        model_path: &Path,
        intra_threads: usize,
        precision: PrecisionMode,
        memory: MemoryMode,
    ) -> Result<Self, TranslatorError> {
        let config = InferenceConfig::new()
            .with_threads(intra_threads as i32)
            .with_precision(precision)
            .with_memory(memory);
        let engine = InferenceEngine::from_file(model_path, Some(config)).map_err(|error| {
            TranslatorError::new(
                TranslatorErrorKind::Internal,
                format!(
                    "failed to load MNN model at {}: {error}",
                    model_path.display()
                ),
            )
        })?;
        Ok(Self { engine })
    }

    pub fn run(
        &self,
        input: &[f32],
        input_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>), TranslatorError> {
        self.engine
            .run_dynamic_raw(input, input_shape)
            .map_err(|e| {
                TranslatorError::new(
                    TranslatorErrorKind::Internal,
                    format!("MNN inference failed: {e}"),
                )
            })
    }
}

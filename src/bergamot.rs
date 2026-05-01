use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use slimt_sys::{
    BlockingService, TokenAlignment as SlimtTokenAlignment, TranslationModel as SlimtModel,
    TranslationWithAlignment as SlimtTranslationWithAlignment,
};

use crate::translate::{TokenAlignment, TranslationMode, TranslationWithAlignment};

const DEFAULT_CACHE_SIZE: usize = 8192;
const DEFAULT_WORKERS: usize = 4;

/// Filesystem locations for a single translation direction's model assets.
///
/// Slimt expects one vocabulary file; bergamot models in our catalog ship a
/// single shared vocabulary, so callers pass the same path for source and
/// target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub vocabulary: PathBuf,
    pub shortlist: PathBuf,
}

pub struct BergamotEngine {
    service: BlockingService,
    models: HashMap<String, SlimtModel>,
}

impl BergamotEngine {
    pub fn new() -> Self {
        let workers = std::env::var("TRANSLATOR_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_WORKERS);
        Self {
            service: BlockingService::with_workers(workers, DEFAULT_CACHE_SIZE),
            models: HashMap::new(),
        }
    }

    pub fn load_model_into_cache(&mut self, paths: &ModelPaths, key: &str) -> Result<(), String> {
        if self.models.contains_key(key) {
            return Ok(());
        }

        let model = catch_slimt_panic(|| {
            SlimtModel::new(&paths.model, &paths.vocabulary, &paths.shortlist, None)
        })?;
        self.models.insert(key.to_string(), model);
        Ok(())
    }

    pub fn evict(&mut self, key: &str) {
        self.models.remove(key);
    }

    pub fn evict_involving(&mut self, language_code: &str) {
        let needle_from = format!("{language_code}-");
        let needle_to = format!("-{language_code}");
        self.models
            .retain(|key, _| !key.starts_with(&needle_from) && !key.ends_with(&needle_to));
    }

    pub fn translate_multiple(
        &self,
        inputs: &[String],
        key: &str,
        mode: TranslationMode,
    ) -> Result<Vec<String>, String> {
        let model = self.model(key)?;
        let refs = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        let html = matches!(mode, TranslationMode::Html);
        catch_slimt_panic(|| Ok(self.service.translate(model, &refs, html)))
    }

    pub fn translate_multiple_with_alignment(
        &self,
        inputs: &[String],
        key: &str,
    ) -> Result<Vec<TranslationWithAlignment>, String> {
        let model = self.model(key)?;
        let refs = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        catch_slimt_panic(|| {
            Ok(self
                .service
                .translate_with_alignment(model, &refs)
                .into_iter()
                .map(map_alignment_result)
                .collect())
        })
    }

    pub fn pivot_multiple(
        &self,
        first_key: &str,
        second_key: &str,
        inputs: &[String],
        mode: TranslationMode,
    ) -> Result<Vec<String>, String> {
        let first_model = self.model(first_key)?;
        let second_model = self.model(second_key)?;
        let refs = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        let html = matches!(mode, TranslationMode::Html);
        catch_slimt_panic(|| Ok(self.service.pivot(first_model, second_model, &refs, html)))
    }

    pub fn pivot_multiple_with_alignment(
        &self,
        first_key: &str,
        second_key: &str,
        inputs: &[String],
    ) -> Result<Vec<TranslationWithAlignment>, String> {
        let first_model = self.model(first_key)?;
        let second_model = self.model(second_key)?;
        let refs = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        catch_slimt_panic(|| {
            Ok(self
                .service
                .pivot_with_alignment(first_model, second_model, &refs)
                .into_iter()
                .map(map_alignment_result)
                .collect())
        })
    }

    pub fn clear(&mut self) {
        self.models.clear();
    }

    fn model(&self, key: &str) -> Result<&SlimtModel, String> {
        self.models
            .get(key)
            .ok_or_else(|| format!("Model not loaded for key: {key}"))
    }
}

impl Default for BergamotEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn map_alignment_result(result: SlimtTranslationWithAlignment) -> TranslationWithAlignment {
    TranslationWithAlignment {
        source_text: result.source,
        translated_text: result.target,
        alignments: result.alignments.into_iter().map(map_alignment).collect(),
    }
}

fn map_alignment(alignment: SlimtTokenAlignment) -> TokenAlignment {
    TokenAlignment {
        src_begin: alignment.src_begin as u64,
        src_end: alignment.src_end as u64,
        tgt_begin: alignment.tgt_begin as u64,
        tgt_end: alignment.tgt_end as u64,
    }
}

fn catch_slimt_panic<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    catch_unwind(AssertUnwindSafe(f)).map_err(panic_to_string)?
}

fn panic_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "slimt panicked".to_string()
    }
}

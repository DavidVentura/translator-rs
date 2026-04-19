use std::path::Path;

use crate::api::{LanguageCode, TranslatorError};
use crate::bergamot::BergamotEngine;
use crate::catalog::CatalogSnapshot;
use crate::routing::{MixedTextTranslationResult, translate_mixed_texts_in_snapshot};
use crate::styled::{
    OverlayScreenshot, StructuredTranslationResult, StyledFragment,
    translate_structured_fragments_in_snapshot,
};

pub struct Translator<'a> {
    engine: &'a mut BergamotEngine,
    snapshot: &'a CatalogSnapshot,
}

impl<'a> Translator<'a> {
    pub fn new(engine: &'a mut BergamotEngine, snapshot: &'a CatalogSnapshot) -> Self {
        Self { engine, snapshot }
    }

    pub fn warm(
        &mut self,
        from_code: &LanguageCode,
        to_code: &LanguageCode,
    ) -> Result<(), TranslatorError> {
        let plan = resolve_translation_plan_in_snapshot(
            self.snapshot,
            from_code.as_str(),
            to_code.as_str(),
        )
        .ok_or_else(|| {
            TranslatorError::missing_asset(format!(
                "translation pack not installed for {}->{}",
                from_code.as_str(),
                to_code.as_str()
            ))
        })?;
        ensure_plan_loaded(self.engine, &plan).map_err(TranslatorError::translation)
    }

    pub fn translate_text(
        &mut self,
        from_code: &LanguageCode,
        to_code: &LanguageCode,
        text: &str,
    ) -> Result<String, TranslatorError> {
        let normalized = text.trim();
        if normalized.is_empty() {
            return Ok(String::new());
        }
        if from_code == to_code || normalized.parse::<f32>().is_ok() {
            return Ok(normalized.to_string());
        }

        let lines = normalized
            .split('\n')
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let non_empty_indices = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (!line.trim().is_empty()).then_some(index))
            .collect::<Vec<_>>();
        if non_empty_indices.is_empty() {
            return Ok(String::new());
        }

        let texts_to_translate = non_empty_indices
            .iter()
            .map(|&index| lines[index].clone())
            .collect::<Vec<_>>();
        let translated = self.translate_texts(from_code, to_code, &texts_to_translate)?;

        let mut merged = lines;
        for (index, translated_text) in non_empty_indices.into_iter().zip(translated.into_iter()) {
            merged[index] = translated_text;
        }
        Ok(merged.join("\n"))
    }

    pub(crate) fn translate_texts(
        &mut self,
        from_code: &LanguageCode,
        to_code: &LanguageCode,
        texts: &[String],
    ) -> Result<Vec<String>, TranslatorError> {
        let plan = resolve_translation_plan_in_snapshot(
            self.snapshot,
            from_code.as_str(),
            to_code.as_str(),
        )
        .ok_or_else(|| {
            TranslatorError::missing_asset(format!(
                "translation pack not installed for {}->{}",
                from_code.as_str(),
                to_code.as_str()
            ))
        })?;
        execute_translation_plan(self.engine, &plan, texts).map_err(TranslatorError::translation)
    }

    pub fn translate_mixed_texts(
        &mut self,
        inputs: &[String],
        forced_source_code: Option<&LanguageCode>,
        target_code: &LanguageCode,
        available_language_codes: &[LanguageCode],
    ) -> Result<MixedTextTranslationResult, TranslatorError> {
        let available_language_codes = available_language_codes
            .iter()
            .map(|code| code.as_str().to_string())
            .collect::<Vec<_>>();
        translate_mixed_texts_in_snapshot(
            self.engine,
            self.snapshot,
            inputs,
            forced_source_code.map(LanguageCode::as_str),
            target_code.as_str(),
            &available_language_codes,
        )
        .map_err(TranslatorError::translation)
    }

    pub fn translate_structured_fragments(
        &mut self,
        fragments: &[StyledFragment],
        forced_source_code: Option<&LanguageCode>,
        target_code: &LanguageCode,
        available_language_codes: &[LanguageCode],
        screenshot: Option<&OverlayScreenshot>,
        background_mode: crate::BackgroundMode,
    ) -> Result<StructuredTranslationResult, TranslatorError> {
        let available_language_codes = available_language_codes
            .iter()
            .map(|code| code.as_str().to_string())
            .collect::<Vec<_>>();
        translate_structured_fragments_in_snapshot(
            self.engine,
            self.snapshot,
            fragments,
            forced_source_code.map(LanguageCode::as_str),
            target_code.as_str(),
            &available_language_codes,
            screenshot,
            background_mode,
        )
        .map_err(TranslatorError::translation)
    }

    pub(crate) fn translate_texts_with_alignment(
        &mut self,
        from_code: &LanguageCode,
        to_code: &LanguageCode,
        texts: &[String],
    ) -> Result<Option<Vec<TranslationWithAlignment>>, TranslatorError> {
        let Some(plan) = resolve_translation_plan_in_snapshot(
            self.snapshot,
            from_code.as_str(),
            to_code.as_str(),
        ) else {
            return Ok(None);
        };
        execute_translation_plan_with_alignment(self.engine, &plan, texts)
            .map(Some)
            .map_err(TranslatorError::translation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TokenAlignment {
    pub src_begin: u64,
    pub src_end: u64,
    pub tgt_begin: u64,
    pub tgt_end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct TranslationWithAlignment {
    pub source_text: String,
    pub translated_text: String,
    pub alignments: Vec<TokenAlignment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranslationStep {
    pub from_code: String,
    pub to_code: String,
    pub cache_key: String,
    pub config: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct TranslationPlan {
    pub steps: Vec<TranslationStep>,
}

fn absolute_install_path(base_dir: &str, install_path: &str) -> String {
    Path::new(base_dir)
        .join(install_path)
        .to_string_lossy()
        .into_owned()
}

fn build_bergamot_config(base_dir: &str, step: &crate::language::LanguageDirection) -> String {
    let model_path = absolute_install_path(base_dir, &step.model.path);
    let src_vocab_path = absolute_install_path(base_dir, &step.src_vocab.path);
    let tgt_vocab_path = absolute_install_path(base_dir, &step.tgt_vocab.path);

    format!(
        "models:\n  - {model_path}\n\
vocabs:\n  - {src_vocab_path}\n  - {tgt_vocab_path}\n\
beam-size: 1\n\
normalize: 1.0\n\
word-penalty: 0\n\
max-length-break: 128\n\
mini-batch-words: 1024\n\
max-length-factor: 2.0\n\
skip-cost: true\n\
cpu-threads: 1\n\
quiet: true\n\
quiet-translation: true\n\
gemm-precision: int8shiftAlphaAll\n\
alignment: soft\n"
    )
}

fn cache_key(from_code: &str, to_code: &str) -> String {
    format!("{from_code}-{to_code}")
}

pub(crate) fn resolve_translation_plan_in_snapshot(
    snapshot: &CatalogSnapshot,
    from_code: &str,
    to_code: &str,
) -> Option<TranslationPlan> {
    if from_code == to_code {
        return Some(TranslationPlan::default());
    }

    let step = |from: &str, to: &str| {
        let pack_id = snapshot.catalog.translation_pack_id(from, to)?;
        let status = snapshot.pack_statuses.get(&pack_id)?;
        if !status.installed {
            return None;
        }
        let direction = snapshot
            .catalog
            .translation_direction(&LanguageCode::from(from), &LanguageCode::from(to))?;
        Some(TranslationStep {
            from_code: from.to_string(),
            to_code: to.to_string(),
            cache_key: cache_key(from, to),
            config: build_bergamot_config(&snapshot.base_dir, &direction),
        })
    };

    let steps = if from_code == "en" {
        vec![step("en", to_code)?]
    } else if to_code == "en" {
        vec![step(from_code, "en")?]
    } else {
        vec![step(from_code, "en")?, step("en", to_code)?]
    };

    Some(TranslationPlan { steps })
}

pub(crate) fn execute_translation_plan(
    engine: &mut BergamotEngine,
    plan: &TranslationPlan,
    texts: &[String],
) -> Result<Vec<String>, String> {
    ensure_plan_loaded(engine, plan)?;
    match plan.steps.as_slice() {
        [step] => engine.translate_multiple(texts, &step.cache_key),
        [first, second] => engine.pivot_multiple(&first.cache_key, &second.cache_key, texts),
        _ => Ok(Vec::new()),
    }
}

pub(crate) fn execute_translation_plan_with_alignment(
    engine: &mut BergamotEngine,
    plan: &TranslationPlan,
    texts: &[String],
) -> Result<Vec<TranslationWithAlignment>, String> {
    ensure_plan_loaded(engine, plan)?;
    match plan.steps.as_slice() {
        [step] => engine.translate_multiple_with_alignment(texts, &step.cache_key),
        [first, second] => {
            engine.pivot_multiple_with_alignment(&first.cache_key, &second.cache_key, texts)
        }
        _ => Ok(Vec::new()),
    }
}

fn ensure_plan_loaded(engine: &mut BergamotEngine, plan: &TranslationPlan) -> Result<(), String> {
    for step in &plan.steps {
        engine.load_model_into_cache(&step.config, &step.cache_key)?;
    }
    Ok(())
}

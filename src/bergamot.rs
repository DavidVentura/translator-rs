use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use slimt_sys::{
    BlockingService, TranslationModel as SlimtModel,
    TranslationWithAlignment as SlimtTranslationWithAlignment,
};

use crate::sentence_split::split_sentences;
use crate::translate::{TokenAlignment, TranslationWithAlignment};

const DEFAULT_CACHE_SIZE: usize = 8192;
const DEFAULT_WORKERS: usize = 4;

/// Filesystem locations for a single translation direction's model assets.
///
/// Most bergamot models in our catalog ship a single shared vocabulary
/// (`vocab.*.spm`). Mozilla's CJK pairs (`en-zh`, `en-ja`, `en-ko`,
/// `en-zh_hant`, `zh_hant-en`) instead ship a `srcvocab.*.spm` +
/// `trgvocab.*.spm` pair and have separate `encoder_Wemb` / `decoder_Wemb`
/// tensors in the model file. For those, set `vocabulary` to the source vocab
/// and `target_vocabulary` to the target vocab; slimt routes them to the
/// right embedding tables. For shared-vocab models leave
/// `target_vocabulary = None` and slimt reuses `vocabulary` for both sides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub vocabulary: PathBuf,
    pub shortlist: PathBuf,
    pub target_vocabulary: Option<PathBuf>,
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
            SlimtModel::with_arch_and_target_vocab(
                &paths.model,
                &paths.vocabulary,
                &paths.shortlist,
                None,
                slimt_sys::ModelArch::default(),
                paths.target_vocabulary.as_deref(),
            )
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

    pub fn translate_multiple(&self, inputs: &[String], key: &str) -> Result<Vec<String>, String> {
        let model = self.model(key)?;
        self.translate_split(inputs, |sentences| self.service.translate(model, sentences))
    }

    pub fn translate_multiple_with_alignment(
        &self,
        inputs: &[String],
        key: &str,
    ) -> Result<Vec<TranslationWithAlignment>, String> {
        let model = self.model(key)?;
        self.translate_split_with_alignment(inputs, |sentences| {
            self.service.translate_with_alignment(model, sentences)
        })
    }

    pub fn pivot_multiple(
        &self,
        first_key: &str,
        second_key: &str,
        inputs: &[String],
    ) -> Result<Vec<String>, String> {
        let first_model = self.model(first_key)?;
        let second_model = self.model(second_key)?;
        self.translate_split(inputs, |sentences| {
            self.service.pivot(first_model, second_model, sentences)
        })
    }

    pub fn pivot_multiple_with_alignment(
        &self,
        first_key: &str,
        second_key: &str,
        inputs: &[String],
    ) -> Result<Vec<TranslationWithAlignment>, String> {
        let first_model = self.model(first_key)?;
        let second_model = self.model(second_key)?;
        self.translate_split_with_alignment(inputs, |sentences| {
            self.service
                .pivot_with_alignment(first_model, second_model, sentences)
        })
    }

    /// Sentence-split each input, batch all sentences in one slimt call,
    /// rejoin per input. The model is trained on per-sentence inputs;
    /// feeding it whole paragraphs causes greedy decoding to occasionally
    /// duplicate or drop spans (see the Pilgrima.ge regression). Splitting
    /// here also lets slimt's batcher pack sentences from different inputs
    /// together for better throughput.
    fn translate_split(
        &self,
        inputs: &[String],
        translate: impl FnOnce(&[&str]) -> Vec<String>,
    ) -> Result<Vec<String>, String> {
        let plan = SplitPlan::build(inputs);
        if plan.sentences.is_empty() {
            return Ok(inputs.iter().map(|_| String::new()).collect());
        }
        let refs: Vec<&str> = plan.sentences.iter().map(String::as_str).collect();
        let translated = catch_slimt_panic(|| Ok(translate(&refs)))?;
        Ok(plan.recombine_strings(&translated))
    }

    fn translate_split_with_alignment(
        &self,
        inputs: &[String],
        translate: impl FnOnce(&[&str]) -> Vec<SlimtTranslationWithAlignment>,
    ) -> Result<Vec<TranslationWithAlignment>, String> {
        let plan = SplitPlan::build(inputs);
        if plan.sentences.is_empty() {
            return Ok(inputs
                .iter()
                .map(|s| TranslationWithAlignment {
                    source_text: s.clone(),
                    translated_text: String::new(),
                    alignments: Vec::new(),
                })
                .collect());
        }
        let refs: Vec<&str> = plan.sentences.iter().map(String::as_str).collect();
        let raw = catch_slimt_panic(|| Ok(translate(&refs)))?;
        Ok(plan.recombine_with_alignment(inputs, raw))
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

/// Bookkeeping for sentence-level pre-splitting. For each Rust-level input
/// we record where its sentences land in a flat batch passed to slimt; the
/// recombine step uses that to stitch translations and alignment offsets
/// back into one per-input result.
struct SplitPlan {
    /// Flat list of sentences sent to slimt, in order.
    sentences: Vec<String>,
    /// For each Rust-level input: the range of indices into `sentences`
    /// occupied by that input's pieces, plus the source-side char offset
    /// of each sentence within the original input. The char offset lets
    /// us shift per-sentence alignment src ranges back into the original
    /// input's coordinate space.
    inputs: Vec<InputPieces>,
}

struct InputPieces {
    /// Half-open range into `SplitPlan::sentences` belonging to this input.
    range: std::ops::Range<usize>,
    /// `src_char_offsets[k]` is the char index (within the original input)
    /// at which sentence `range.start + k` begins. Used to re-base
    /// per-sentence src alignments.
    src_char_offsets: Vec<u64>,
}

impl SplitPlan {
    fn build(inputs: &[String]) -> Self {
        let mut sentences: Vec<String> = Vec::new();
        let mut input_meta = Vec::with_capacity(inputs.len());
        for input in inputs {
            let start = sentences.len();
            let mut src_char_offsets = Vec::new();
            for slice in split_sentences(input) {
                let byte_offset = slice.as_ptr() as usize - input.as_ptr() as usize;
                let char_offset = input[..byte_offset].chars().count() as u64;
                src_char_offsets.push(char_offset);
                sentences.push(slice.to_string());
            }
            let end = sentences.len();
            input_meta.push(InputPieces {
                range: start..end,
                src_char_offsets,
            });
        }
        SplitPlan {
            sentences,
            inputs: input_meta,
        }
    }

    fn recombine_strings(&self, translated: &[String]) -> Vec<String> {
        self.inputs
            .iter()
            .map(|input| {
                if input.range.is_empty() {
                    return String::new();
                }
                translated[input.range.clone()].join(" ")
            })
            .collect()
    }

    fn recombine_with_alignment(
        &self,
        inputs: &[String],
        raw: Vec<SlimtTranslationWithAlignment>,
    ) -> Vec<TranslationWithAlignment> {
        self.inputs
            .iter()
            .zip(inputs.iter())
            .map(|(input, original)| {
                if input.range.is_empty() {
                    return TranslationWithAlignment {
                        source_text: original.clone(),
                        translated_text: String::new(),
                        alignments: Vec::new(),
                    };
                }
                let mut combined_target = String::new();
                let mut alignments = Vec::new();
                let mut tgt_char_cursor: u64 = 0;
                for (i, slimt_idx) in input.range.clone().enumerate() {
                    let res = &raw[slimt_idx];
                    let src_offset = input.src_char_offsets[i];
                    if i > 0 {
                        // Insert a single space between sentences in the
                        // combined target to match `recombine_strings`.
                        // The space char is "owned" by no alignment; the
                        // HTML pipeline's gap-fill uses neighbouring
                        // alignments to attribute it.
                        combined_target.push(' ');
                        tgt_char_cursor += 1;
                    }
                    combined_target.push_str(&res.target);
                    for a in &res.alignments {
                        alignments.push(TokenAlignment {
                            src_begin: src_offset + a.src_begin as u64,
                            src_end: src_offset + a.src_end as u64,
                            tgt_begin: tgt_char_cursor + a.tgt_begin as u64,
                            tgt_end: tgt_char_cursor + a.tgt_end as u64,
                        });
                    }
                    tgt_char_cursor += res.target.chars().count() as u64;
                }
                TranslationWithAlignment {
                    source_text: original.clone(),
                    translated_text: combined_target,
                    alignments,
                }
            })
            .collect()
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

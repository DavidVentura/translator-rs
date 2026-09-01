# Add a language pair

This is the canonical guide for adding and training an `en↔X` pair. It covers
language-specific checks, data preparation, teacher selection, distillation,
student training, evaluation, and packaging.

Use [`NOTES.infra.md`](NOTES.infra.md) for machine, container, and Vast.ai
details. Keep measurements from a particular pair in its findings or run record.

## Workflow

Complete the stages in order:

1. Define the language, production inputs, and release requirements.
2. Inspect writing-system, normalization, segmentation, and tokenization behavior.
3. Inventory and qualify the available corpora.
4. Gate a teacher independently for each direction.
5. Prepare cleaned data, held-out data, and the joint vocabulary.
6. Decode the KD source and select or filter teacher outputs.
7. Align the selected corpus and train the student.
8. Evaluate the quantized artifact on every relevant slice.
9. Package the model and complete the release checks.

Before paid compute, record the language profile, corpus inventory, contamination
checks, teacher decision, evaluation sets, and budget. Stop when a required gate
fails.

## 1. Define the pair

Record:

- the dataset language code and any teacher-specific language codes
- the two translation directions
- the target script and any script variants
- the definition of production inputs
- the available native-speaker review
- the model size, latency, and package constraints

Use separate decisions for `en→X` and `X→en`. Data availability, teacher
quality, morphology, and evaluation behavior can differ by direction.

### Production inputs

Define production inputs as the range of text encountered during ordinary daily
use. This is deliberately broad. It can include a short sign or menu label, a
casual message, a Wikipedia article, or a government letter.

This definition describes coverage rather than priority. No category is assumed
to dominate the others. Evaluate both short and long inputs explicitly because a
model can handle longer prose while treating short input as out of distribution,
then looping, copying, or hallucinating.

## 2. Inspect the language behavior

### Writing system and casing

Check whether the language has case distinctions and whether the script contains
case variants or display forms. Read the actual OpusTrainer modifiers before
choosing augmentation settings.

For a caseless language or a script whose transformed characters leave the
training vocabulary, disable the affected case modifiers in a pair-specific
configuration. Verify the training log reports the intended modifier weights.

Check normalization at the translation input boundary. Preserve the original OCR
text for copy and per-word paths when normalization is only required by MT.

### Sentence segmentation

Run the runtime splitter on representative text containing at least two
sentences. Check full stops, question marks, exclamation marks, abbreviations,
ellipses, and non-breaking prefixes.

The splitter used for corpus preparation must match the runtime splitter. Split
sentences before the KD draw so source and reference rows remain aligned and the
draw remains reproducible.

### Vocabulary

Measure teacher output tokens per character for both directions. Use the result
when estimating KD cost and decode time.

Train a pair-specific joint SentencePiece vocabulary. Check piece count, unknown
tokens, encode/decode round trips, and fertility for each side. A low unknown rate
does not prove that the vocabulary represents the script efficiently because byte
fallback can hide a missing script.

### Evaluation behavior

Choose metrics after inspecting the language. Morphology, tokenization, word
order, and script properties affect the relationship between surface metrics and
translation quality.

Use chrF and spBLEU where they provide useful comparisons. Use COMET only with
the language-coverage and calibration limitations understood. Treat metrics as
system-ranking tools when human-judgment coverage is limited.

Create production-input probes for signs, menus, warnings, dosages, entities,
short inputs, and content words. Include adversarial cases for numbers,
negations, copying, repetition, and length blowup. Review the translations by
kind rather than reducing the probe set to one score.

Evaluate the teacher on short inputs as a separate slice. Use one-word, short
sentence, conversational, and other production-input examples. Check whether the
teacher preserves meaning, stops at the right point, and handles entities,
numbers, and negation. A news benchmark does not establish short-input quality.

## 3. Inventory and qualify the data

Start with the OPUS inventory:

```text
curl -sL "https://opus.nlpl.eu/opusapi/?source=en&target=<lang>&preprocessing=moses&version=latest"
```

Compare the result with the corpus allowlist in `registers.py`. Record the
available pair counts, the largest contributors, missing sources, unique source
and target lines, and the amount of usable human text.

The standard preparation filters remove formatting artifacts, malformed wiki
markup, invalid character or word ratios, number and URL mismatches, excessive
word lengths, and wrong-language rows. Inspect the filter implementation before
changing a default, and report what each filter removes by corpus and register.

Qualify the non-English source independently of pair alignment. A mined pair can
be well aligned while its non-English side is itself an upstream machine
translation: correctly encoded and spaced, yet calqued, ungrammatical, or
semantically unreliable. Mechanical rules can remove invalid encoding, broken
script mixtures, and other surface damage, but cannot establish that apparently
fluent text was written naturally in the language.

Draw a uniform sample from each important corpus and register before committing
to KD. Have a native reviewer, or a capable model with native review of its
calibration set, classify the source side for native quality and translationese.
Use that sample to estimate the dirty rate and decide whether the corpus remains
worth keeping; do not infer the rate from rows selected by a proposed filter. If
trusted native monolingual data is available, prefer decoding it with the teacher
to trying to repair a corpus whose source was machine-generated upstream.

When no replacement corpus is available, train an acceptability/translationese
classifier to rank the existing source. It needs reviewed clean and degraded
examples: a clean-only language model may help mine suspicious candidates, but
will also rank names, rare terminology, and legitimate register differences as
unusual. Calibrate the classifier's cutoff on a separate representative reviewed
sample, then retain only the confidence range whose false-positive cost is
acceptable.

Measure a defect rate before building a filter for it. Estimate it from a
uniform random sample judged directly. A rate read off the filter's own output
measures the filter. Keep any deliberately skewed sample separate and use it only
to evaluate a proposed cut.

Character and token models detect orthographic damage. They do not detect a
source that is fluent but wrong, which is correctly spelled and scores near the
median of its length. Detecting that requires the reference side of the pair.
State which of the two an instrument detects, and do not let it be cited as the
other.

Where a defect has no negative class to train against, emit a score and calibrate
it against labelled lines rather than thresholding it directly. An absolute
perplexity cut rejects proper nouns, loanwords and short lines, and a
length-normalised score ranked globally is a short-line filter. Rank within
length buckets and stratify any labelling set by length.

Record the licence of every source alongside its size, and decide eligibility
before measuring quality. The app is non-commercial, so CC BY-NC sources are
usable.

Classify corpora by role and register. Keep sentence-level prose, UI strings,
named-entity data, subtitles, spoken text, and religious or historical text
separate when deciding what to train on.

### KD scale and composition

Plan for roughly 4–10 million usable KD sentence pairs per direction. Choose the
point in this range from teacher quality, token cost, language availability, and
student behavior. Record the reason when the available data or teacher makes a
different scale appropriate.

Vary the source by corpus and register. Track the share contributed by each
source, and check whether one news or wiki crawl dominates the pool. Most
teachers are strongest on news and web prose, so a large crawl pool can still
leave the student weak on short and conversational inputs. The composition of the
pool is part of the training design, not a property to inspect only after a run.

If OpenSubtitles is available, use a cleaned portion to add conversational and
short inputs. Subtitle data often combines several sentences and contains
formatting artifacts. Run `split_sentences.py` before sampling, alignment, or KD
decoding, and verify that its behavior matches the runtime splitter. Inspect
abbreviations, ellipses, and subtitle markers specifically.

If suitable conversational or short bitext is unavailable, generate a small,
explicitly labeled finetune set with a capable low-cost SOTA model. For example,
use `gpt-5.6-luna-low` when it is available and meets the current quality and
cost requirements. Generate from production inputs, deduplicate by source,
review the outputs, and record the model and prompt version. Keep this set
separate from human bitext and use it to address the short-input gap. It does not
replace the KD corpus scale requirement.

Use these corpus roles:

- **KD source:** the source column used for teacher translation. The teacher
  creates the target, so source-side quality is the primary concern.
- **Finetune data:** aligned source and target text. Use curated or high-confidence
  human bitext, or explicitly labeled SOTA-generated data when human data is
  unavailable. Semantic misalignment directly teaches incorrect mappings.
- **Validation data:** held-out two-column pairs in the direction being trained.
- **Evaluation data:** held-out references and production-input probes that are
  excluded from training by content hash.

Prepare the KD source and its original reference column together. Use
`build_kd_source.sh` rather than independently sorting and sampling the two
columns.

Bicleaner measures source-reference alignment. Use its score to gate
reference-based n-best selection and to salvage high-confidence human bitext for
finetuning or the backward model. It does not replace source-side cleaning for
the KD pool, because the teacher regenerates KD targets.

## 4. Check contamination

Check the following classes before accepting a corpus:

1. Sibling languages that share the script.
2. Encoding or legacy-font corruption that remains inside the expected Unicode
   block.
3. Archaic, liturgical, or otherwise unsuitable registers.
4. Register imbalance introduced by filtering.

Use script and character checks, language identification, frequency analysis,
and character or token models as appropriate. A keep-only rule is unsafe when
absence of a marker does not prove language membership. Prefer drop-only rules
for ambiguous sibling-language signals.

Validate every filter on known-clean data. Record its false-positive rate and the
retained and removed counts by corpus and register. If filtering removes the only
conversational or short-input source, decide whether the loss is acceptable before
training.

Keep a corrupt target row as a KD source when the source is usable and the teacher
will regenerate the target. Do not use that row as human finetune data.

## 5. Gate the teacher

Run a loading smoke test before interpreting a poor score. Reproduce the model
card example, pin the model and library versions, and confirm the tokenizer,
language codes, direction, and output format.

Evaluate every candidate in both directions when both directions are intended for
production use. Use FLORES for a common reference point, then use the production-input
check set, held-out register slices, and the probe set. A teacher that passes a
news benchmark can still fail on short signs, entities, numbers, or negations.

Choose a teacher per direction using the complete evidence. Do not apply one
absolute chrF threshold to languages whose surface metrics are known to differ.
Record the teacher, model version, decoding settings, evaluation inputs, scores,
and review decision.

If references were generated by a model family used in training, check for style
agreement. A gain on generated references must also appear on human-referenced
slices and in manual review.

## 6. Prepare the training artifacts

Run the data preparation script on a local CPU-only host when available. Use the
project's uv-managed Python environment:

```text
prep_data.py --lang <lang> --workdir <workdir> --jobs <n>
```

The first run trains the joint vocabulary. Use `--skip-spm` only when reusing a
verified vocabulary for a separate data role. The vocabulary passed to Marian
must have the `.spm` extension.

Create, verify, and retain:

- cleaned pair data
- the joint vocabulary
- the KD source and matching reference column
- the held-out validation file
- the source-only devtest file used by quantization
- content hashes for excluded evaluation rows
- any bicleaner scores or contamination decisions used later

Run the vocabulary checks before renting a training box. A vocabulary failure is
cheaper to correct before alignment and KD artifacts depend on it.

## 7. Decode and select KD data

Shard the KD source before teacher decoding. Keep shard order, source rows,
references, scores, and output rows together.

Select a teacher with a supported gate and decoding path. The pipeline has used
NLLB, Hy-MT2, and OPUS-MT teachers. NLLB and OPUS-MT use `distill_data.py`; a
causal language model such as Hy-MT2 uses the corresponding vLLM decoding path.

For an NLLB or OPUS-MT teacher, use `distill_data.py` with the corresponding
teacher language codes:

```text
distill_data.py --model <model> --src <shard> --out <targets> \
  --src-lang <src_code> --tgt-lang <tgt_code> \
  --beam <beam> --nbest <n>
```

Use n-best selection only when the source-reference pair is trustworthy. Gate
reference-based selection with a pair-quality score. Use rank one when the
reference is below the selection gate. Keep the reference-selection threshold
separate from the stricter threshold used to salvage human finetune data.

Use the backward RNN to score teacher output against the original source when
the pipeline includes the ce-filter. The backward model must be trained from
independent human or high-confidence aligned bitext. Drop only the configured
worst fraction, preserve the row order, and record the filter coverage.

Do not assume that a moved n-best hypothesis improved the translation. Compare
the resulting student against the rank-one control using all evaluation slices
and the probes.

## 8. Align and train the student

Remove empty or zero-token pairs before alignment. Run both alignment directions,
symmetrize them, and fail the step when either direction contains an unexpected
number of empty alignments.

Create the guided-alignment TSV with:

```text
align.sh <source> <target> <out_dir> <tools_dir>
```

Run the CPU-only stages on a local host when available before renting a GPU. Use
Docker for binaries that require an isolated system environment. The production
student uses the base-memory configuration unless a documented experiment selects
another size.

Train with `train_student.sh` and a two-column validation file. Keep the best
checkpoint and stop according to validation behavior. Use the pair-specific
OpusTrainer configuration when script casing, punctuation, or normalization
requires it.

Use guided alignment for the production student. The OpusTrainer stream supplies
the augmented three-column corpus to Marian, and the model's alignment output is
used for format and bold transfer. Verify that the training configuration,
vocabulary, and alignment mode are the intended pair-specific versions.

Treat finetuning as an explicit experiment. Use curated or high-confidence human
bitext, or explicitly labeled SOTA-generated data when suitable human data is
unavailable. Deduplicate generated inputs, record the model and prompt version,
evaluate every slice, and retain the base checkpoint as the comparison.
Finetuning can improve the intended production slice while reducing performance
on other registers.

Run the multi-GPU efficiency check before committing to a multi-GPU training job.
The measured throughput must be evaluated on a representative corpus and include
the effective batch and learning-rate settings.

## 9. Quantize, evaluate, and package

Quantize the selected checkpoint with the browsermt conversion path:

```text
quantize_export.sh <model.npz> <vocab.spm> <devtest.src> <out_dir>
```

Build the shortlist from SentencePiece subwords with:

```text
shortlist.sh <source> <target> <vocab.spm> <out_dir> <tools_dir>
```

Benchmark the quantized model without a shortlist first. Then benchmark the
shortlisted package. Compare the model and package against the teacher, the
previous package, held-out slices, and probes.

Package only after the artifact passes the release checks. Use a new package
directory for each model update and verify hashes, sizes, vocabulary, shortlist,
and index references before publishing.

## Completion checklist

A language pair is ready for release when:

- the language profile and pair-specific configuration are recorded
- corpus roles, contamination checks, and held-out hashes are recorded
- the KD corpus has an appropriate scale and a measured source/register mix
- short-input behavior has been evaluated for the teacher and student
- the vocabulary passes script, round-trip, unknown-token, and fertility checks
- each teacher passes the load, reference, production-input, and probe checks
- KD, alignment, filtering, and training artifacts are reproducible and verified
- the quantized model and shortlist pass independent package tests
- all evaluation slices and probes have been reviewed
- the release decision and remaining limitations are recorded

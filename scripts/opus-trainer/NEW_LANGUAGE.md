# Adding a language: what to check before spending money

`NOTES.md` is the pipeline — which script, in what order, on what hardware. This
file is the part that changes per LANGUAGE and that the pipeline cannot tell you:
what to look at in the data, which properties of the writing system will break a
default, and which of your instruments stop meaning anything.

Worked example throughout: en↔ka (Georgian), 2026-08-30/31. Numbers and evidence
in `ka_findings.md`.

---

## 0. Order of operations

Everything below is cheap and none of it needs a GPU. Do it before renting one.

1. Language properties (§1) — an afternoon of reading, decides several defaults
2. Corpus availability (§2) — one OPUS API call
3. Contamination survey (§3) — the step most likely to save a wasted run
4. Teacher gate (§4) — ~$1 of GPU, decides the whole campaign
5. Only then: prep → KD → align → train

---

## 1. Language properties that change a default

Answer these before touching the pipeline. Each has bitten at least once.

**Is the script unicameral?** A language with no case breaks OpusTrainer's
`UpperCase` AND `TitleCase` modifiers, because both call `.upper()`, which may
map into a *different Unicode block* the SPM never saw. Georgian's Mtavruli
(U+1C90–1CBF) is the case that cost a retrain. Symptom: the student emits
codepoints absent from the training corpus; chrF barely moves, so the metric will
not warn you. Fix: a per-language `configs/opustrainer.student.<lang>.yml` with
both at 0.0, selected via `OPUSTRAINER_CFG`.

Do NOT test this with Python's `.title()`. It is a no-op on some caseless scripts
while OpusTrainer's modifier — `word[0].upper() + word[1:]` — is not. Read the
modifier's source.

The same question decides whether the runtime splits the language at all. A
sentence splitter that guards `.` with an uppercase check silently never splits a
caseless script, so the model receives whole paragraphs while trained on single
sentences. Ours did this for Georgian, Hebrew, Arabic, Devanagari, Bengali, Thai
and Japanese; only `!` and `?` split, which hides it in dialogue. Test the
runtime splitter on a two-sentence string in the new language before trusting
either the corpus prep or the inference path, and see `ka_findings.md` §17 for
the two traps in the non-breaking-prefix list that follow from fixing it.

**Does the script have a separate "display" form?** Caps-for-signage,
titling variants, presentation forms. If OCR emits it and web text does not, the
teacher has never seen it. Decide where to normalise — we chose the MT input call
site, so the OCR text keeps what was on the page for the copy/per-word paths.

**How token-efficient is the language in the teacher's vocabulary?** Measure it:
mean output tokens ÷ mean output characters. Georgian ran ≈1.0 token/char in
Qwen3's vocab against 2.6 chars/token for English, i.e. 2.6x the decode cost per
unit of text. This directly sets your KD budget and is invisible until you are
paying for it.

**Morphology, and what it does to your metrics.** Agglutinative or
morphologically rich targets depress chrF and *destroy* BLEU: one model scored
BLEU 14.90 / COMET 89.17 on en→ka against BLEU 38.79 / COMET 88.87 on en→de —
same adequacy, 2.6x the BLEU. Consequences:
- absolute chrF is NOT comparable across target languages; a "≥50 chrF" gate rule
  calibrated on Latin-script pairs can reject every teacher that exists
- drop BLEU entirely for such targets
- into-X and X-into-English scores are not comparable either (~9 chrF apart on ka)

**Is COMET actually trained on this language?** `wmt22-comet-da`'s encoder covers
~100 languages but its human-judgment data covers far fewer. On an uncovered
language it is zero-shot: useful for RANKING systems on one test set, not as a
calibrated number. And it is separately blind to meaning inversions — read the
pairs.

**Vocabulary sizing.** A joint SPM splits its budget between the two sides. For a
morphologically rich target, check `check_vocab.py`'s FERTILITY per side, and
read the number rather than only checking the gate passes: the 6.0 default was
calibrated against byte-fallback catastrophe (8.6), not against "adequate but
over-fragmented".

---

## 2. Corpus availability, in one call

```
curl -sL "https://opus.nlpl.eu/opusapi/?source=en&target=<lang>&preprocessing=moses&version=latest"
```
Gives every corpus and its pair count in seconds. What to look for:

- **Total usable bitext.** Shipped pairs have ranged from ~2.9M (sw) to 47M (tl).
- **The mined share.** If one crawl corpus is >70% of the total, your pool is
  effectively that corpus and inherits its defects. en-ka was 76% NLLB.
- **Genuinely clean human text**, which is what finetune material is made of, and
  is always far smaller than the headline. Separate sentence-level prose from UI
  strings and from named-entity pairs — they are not interchangeable.
- **Absences.** Missing CCMatrix/ParaCrawl for a pair is normal and worth knowing
  before you budget on a total that includes them.
- **Unique lines per COLUMN, not just pair count.** The two directions dedup on
  different sides of the same corpus, so they do not have the same amount of
  usable data. en-ka crawl held 10.7M unique English against 6.7M unique
  Georgian, and the ka→en draw ran out at 6.31M after asking for 10M. Measure
  both columns before budgeting either direction.
- **`registers.py` membership is the allowlist.** A corpus absent from that table
  is silently never downloaded. Diff the API's corpus list against the table.

---

## 3. Contamination: the survey that saves a run

Mined corpora for lower-resource languages carry defects that pass every obvious
check. Look for all four before trusting a pool.

**(a) Sibling languages sharing the script.** List the languages that use this
writing system, then check whether your LID can separate them. fastText `lid.176`
has `ka` and `xmf` (Mingrelian) but no Laz or Svan, so those fall through to `ka`
silently. Where a sibling has script characters the main language lacks, that is
a *drop-only* rule — never keep-only, since absence proves nothing.

**(b) Encoding corruption that lands inside the right Unicode block.** The one
that cost the most here. ~80% of the Georgian side of OPUS OpenSubtitles is
South-Slavic text rendered through a legacy 8-bit font table: CP1251 Cyrillic
bytes painted with Georgian glyphs and stored as Georgian codepoints. It passes
codepoint-range checks, "contains the script" checks, and `lid.176`.

How to find it: take the most frequent characters on the target side and check
whether their distribution looks like the language. If a legacy font mapping
exists, it is usually the *traditional alphabet order* of the script aligned onto
some 8-bit codepage's order — derive it and invert it, then LID the result.
Detection ladder that worked (`mojibake_filter.py`): archaic/rare codepoints as a
strong drop, then a Georgian-vs-Slavic character-trigram likelihood ratio where
the Slavic model is trained on the corpus's OWN tier-1 hits, costing nothing.
Demap-then-LID was tried and rejected: LID calls demapped genuine text `ru`,
because it is still Cyrillic.

**(c) Archaic or liturgical register.** Bible and scripture translations are
heavily represented in web crawls for smaller languages, often in a historical
orthography. Correctly aligned, genuinely useless for modern signage. On ka this
was ~30k lines using Old Georgian forms. Detect via archaic codepoints plus a
small marker list of function words the modern language does not use.

**(d) Register imbalance you create by filtering.** Cleaning is not free. Removing
the mojibake took en-ka dialogue from 162,962 to 40,087, because OpenSubtitles
was the ONLY conversational corpus. Check what each filter costs per register,
not just overall. For an en→X direction the loss is often recoverable — the KD
source is the ENGLISH side and the teacher regenerates the target, so a pair whose
only defect is a corrupt target is still a usable source line.

**Always validate a filter on known-clean data before running it on the pool**,
and report the false-positive rate. A tempting heuristic here dropped 1.1% of
clean Georgian because the word "და" (and) demaps to Bulgarian "да".

---

## 4. Gating a teacher

Published scores are not enough, for two reasons this pair demonstrated.

**Reproduce the model card's own example before believing a bad result.** A
teacher that appears catastrophic may be loaded wrong: a config flag defaulted
differently by a newer library version silently broke MADLAD-400 across every
language. `<2pt> I love pizza!` → `Eu adoro pizza!` costs one line and separates
"bad at our language" from "we loaded it wrong". Pin one library version across
every model in a comparison.

**Published anomalies can have mechanical causes worth finding.** MADLAD-400's
unexplained ~15 chrF on en→ka is because 59% of its Georgian output is the same
Slavic mojibake as §3(b) — it trained on the corrupted web text. Unusable, and
filtering does not rescue it, because underneath is fluent off-topic
hallucination.

**Gate on the deployment distribution, not FLORES.** FLORES is clean news and
hides what decides shippability. Build:
- a **check set** of camera-path strings (signs, menus, dosages, prohibitions) —
  no corpus contains these in any language, so they must be authored and
  reviewed. This is the only part that needs a native speaker, and it is the
  highest-value hour anyone spends on the pair.
- **held-out human slices** per register from OPUS (subtitles, spoken, UI) — free,
  real references. Exclude them from training by content hash, and keep the hash
  list next to the corpus.
- the shared reference-free `probes/adversarial.en`, which needs no target
  language at all.

**Beware self-referential references.** If the check set's target side is
model-generated and your finetune data comes from the same model family, a large
gain there may be style agreement. Test it by reading, and by checking whether
the human-referenced slices move in the same direction.

---

## 5. Two things that generalise from training

**A short-string finetune is a fixed trade, not a free win.** Curated
frontier-generated data put the en→ka student 8.67 chrF++ ABOVE its teacher on
signage — but cost −1.5 flores, −1.5 ted, −3.2 ui. Adding long-form and
sentence-splitting to counter that moved everything under a point in both
directions: finetuning on ~120k after 4M shifts the model toward those 120k
whatever is in them. Budget the trade; judge on ALL slices, since the one it was
aimed at will always look good.

**Sentence-split the corpus before the KD draw.** The app splits at inference, so
the model only ever sees one sentence, while ~10% of mined bitext is
multi-sentence. Splitting after alignment means redoing alignment; after the draw
it breaks the pinned artifact. Use `split_sentences.py`, which mirrors the
runtime's non-breaking-prefix list — a bare regex breaks on "Mr." and on ellipsis
and invents a mis-alignment rate that is not there.

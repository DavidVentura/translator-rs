# CPT → SFT: a Uyghur-specialized translation teacher

Status: **PARKED on budget 2026-07-15 — and the CPT-Gemma teacher below is DEMOTED.**

> ⚠️ **STRATEGIC UPDATE (2026-07-15) — read before following the plan below.** The empirical
> work this session (COMET22 gate + a real 2500-pair LoRA-SFT) changed the conclusion. The
> body of this doc argues "build a CPT→SFT 4B teacher and distill it." **We now think that's
> the wrong shape.** Why:
> - COMET22 showed **NLLB Uyghur is not garbage** (0.78-0.88, right at Mozilla's shipped
>   low-resource floor — az 0.837, fa 0.846), and the frontier edge over NLLB is real only on
>   **ug→en dialogue (+0.079)** and **en→ug article (+0.049)**, negligible elsewhere. A 4B
>   teacher only earns its keep if it clearly EXCEEDS NLLB; the 2500-pair SFT landed at exactly
>   NLLB-level (chrF 41.6 ≈ 0.825 COMET) → as a teacher it added nothing.
> - So: **don't build the 4B teacher.** Inject frontier (Sonnet, 0.87 COMET) quality
>   **upstream** into the existing/new NLLB→slimt pipeline (the mozilla-gap levers in
>   `NOTES.md`: extract-best reference / ce-filter) — NOT as a bespoke teacher, NOT as a
>   student-finetune step.
> - **Role assignment per NOTES "TWO TIERS OF HUMAN DATA" (corrects an earlier mis-claim here
>   that a $1-3k Sonnet subset should be the filtering reference — WRONG).** The extract-best /
>   ce-filter references are **Tier B = mined bitext at KD scale, FREE** — our KD source is the
>   bitext source side, so every KD line already carries its own reference; you never generate
>   them, you just stop discarding that column. Quality handled by **bicleaner-gating** (Mozilla's
>   answer), and degradation is graceful — a noisy ref ≈ 1-best, it wastes the opportunity but
>   doesn't poison. The **2.5k Opus en→ug tranche is Tier A** (teacher anchor / the
>   finetune-which-stays / ~1-2k calibration) — thousands-scale, ~400× smaller than a KD
>   reference set; the two tiers don't interchange. **Targeted SOTA (Sonnet) references at ~$1-3k
>   are escalation-ONLY** if uig proves genuinely reference-starved *after* bicleaner-gating the
>   mined refs — never the default; full-KD-scale generated references = $10k+/pair = the wrong way.
> - **Gated on: (a) the concurrent student-distillation-loss fix landing** (Swahili proves the
>   new pipeline → Uyghur inherits it) **and (b) budget** (~$50 spent = ceiling). Until both,
>   this rests at $0. The empirical sections below (frontier gate, COMET, spend map) are the
>   durable output; the "build a teacher" narrative is the superseded reasoning trail.

This is the escalation path for Uyghur, which is teacher-blocked for the normal NLLB/OPUS
distillation pipeline (see `NOTES.md`). Read `NOTES.md` first for the pipeline this feeds into.

## Where we started

Uyghur (`ug`, Perso-Arabic script) is a requested language with **no usable off-the-shelf
teacher**:

- NLLB gate (FLORES devtest, chrF++): en→uig **38.6** (600M) / **43.2** (1.3B); uig→en
  **47.3** / **49.7**. The into-Uyghur direction is below the ~50 ship bar even at 1.3B.
- Hy-MT2 gate (FLORES, chrF++): 7B en→uig **43.0** (merely *ties* NLLB-1.3B), uig→en 51.6.
  Hy-MT2 is Chinese-centric, so its 1.8B is *worse* than NLLB. A bigger/newer general
  translation LLM does not unlock the hard direction — Uyghur is data-starved everywhere.

The normal pipeline (distill NLLB or OPUS-MT into a base-memory slimt student) therefore
caps Uyghur at NLLB's weak quality. The blocker is the **teacher**, not the student arch.

## The measurement that changed the picture

We found **MiLiC-Eval** (PKU PIE Lab), a human-translated en↔ug benchmark with two domains
(article: 912 pairs, dialogue: 673 pairs), scored in chrF++ — a real Uyghur eval beyond
FLORES. We gated **pkupie/gemma-3-4b-ug-cpt** (Gemma-3-4B continued-pretrained on the
Uyghur MC² corpus — a *base* model, no translation training) via 5-shot prompting, against
NLLB, on MiLiC-Eval:

| model | article en→ug | article ug→en | **dialogue en→ug** | **dialogue ug→en** |
|---|---|---|---|---|
| NLLB-600M | 38.77 | 47.31 | 33.63 | 39.06 |
| NLLB-1.3B | **42.99** | **49.86** | 35.82 | 41.95 |
| pkupie CPT (5-shot) | 37.65 | 45.44 | **36.56** | **47.54** |

(NLLB's article numbers reproduce its FLORES numbers, so the harness is sound. NLLB
collapses on dialogue.)

**The domain split is the finding:**

- Formal/article text → NLLB-1.3B wins (it is trained on formal parallel corpora).
- Dialogue/colloquial → the Uyghur-CPT model wins **both directions**, and ug→en by **+5.6**
  over NLLB-1.3B. Even into Uyghur (the "blocked" direction) it edges NLLB on dialogue.

pkupie has **zero translation training** — this is a raw CPT base with few-shot prompting.
That is a floor, not a ceiling. For a phone translation app, dialogue is the domain that
matters most, and a Uyghur-adapted 4B already beats NLLB there.

## The idea

A model *specialized* to Uyghur beats general teachers on the domain that matters, because
the specialization (deep Uyghur language modeling from CPT) lives in the weights and cannot
be back-doored into a tiny student distilled from NLLB. So: build a purpose-built en↔ug
model and use it as the **teacher** for the existing slimt distillation pipeline.

Two stages, well-separated:

- **CPT (continued pre-training)** on monolingual Uyghur → teaches the model the *language*.
  Already done by pkupie on Gemma-3; can be redone on Gemma-4 for a cleaner license.
- **SFT (supervised fine-tuning)** on parallel translation pairs → teaches the *task*
  (translate, in a fixed format, then stop). This is the missing half.

```
Gemma base ──CPT(MC²)──► Uyghur-fluent base ──SFT(pairs)──► en↔ug translator ──distill──► slimt student
             (pkupie did this,      (~$1-3 QLoRA,          (normal pipeline,
              or ~$50 on Gemma-4)    the new step)          NOTES.md)
```

## Why the architecture is sound

- **CPT teaches language, SFT teaches task.** They are different trainings on different data
  (monolingual vs parallel). SFT cannot teach a language the base does not know — on a
  low-resource language the base's ability is the ceiling — which is exactly why the CPT
  step exists and why `raw Gemma-4 + SFT` is weaker than `CPT-Gemma + SFT` for Uyghur.
- **SFT is cheap and small** *because* CPT already did the language learning. ~10-50k pairs
  is enough to teach the task and balance domains; you are not learning Uyghur from 20k
  pairs, you are learning to translate. This matches the community-data scale.
- **Specialization beats generalization on a narrow target.** The result is a scoped-down
  "TranslateGemma for en↔ug" — one pair, SFT-only, LoRA — not the general 160-language
  RL-refined model. For Uyghur that specialist can beat the general TranslateGemma, which
  does not even cover Uyghur in its benchmarked set.
- **Evidence already exists:** the raw CPT base (no SFT) beats NLLB on dialogue today. SFT
  only has to (a) fix the article gap and (b) remove few-shot hallucination.

## The plan

### 1. Base model

- Cheapest: use **pkupie/gemma-3-4b-ug-cpt** as-is (CPT done, free) — inherits the Gemma
  license (Prohibited Use Policy pass-through, awkward for a GPLv3 project).
- Cleaner: redo CPT on **Gemma-4 base (Apache 2.0)** using MC² (~$20-80, ~40 GPU-hours; MC²
  Uyghur is ~150M tokens, model ~4B). Gemma-4 is a stronger base and license-clean.

### 2. SFT data (the recipe)

Mix two domains so the model is good at both:

- **Article (fix pkupie's gap): NLLB-1.3B-generated pairs.** Source text →
  - en→ug: English Wikipedia/news → NLLB → Uyghur.
  - ug→en: **Uyghur MC² / Uyghur Wikipedia → NLLB → English** (this is MC²'s real use here).
- **Dialogue (keep pkupie's win): human pairs.** The community's conversational en↔ug pairs
  (the real lever). For the quick test, OPUS colloquial sets (QED / subtitles) as a stand-in.

Both directions in one training set. Format as instruction examples (Gemma chat template),
loss masked to the response:

```json
{"messages":[{"role":"user","content":"Translate English to Uyghur:\n{en}"},
             {"role":"assistant","content":"{ug}"}]}
{"messages":[{"role":"user","content":"Translate Uyghur to English:\n{ug}"},
             {"role":"assistant","content":"{en}"}]}
```

**Scale: ~10-50k, balanced across domains. NOT 1M.** A 1M NLLB-heavy run makes the model
*imitate* NLLB and washes out the dialogue edge — you would spend money rebuilding an NLLB
clone. SFT saturates fast; the dialogue advantage lives in the CPT weights and survives a
balanced modest SFT, but a huge NLLB-dominated set erodes it. The article ability, learned
by imitating NLLB, approaches NLLB (~43/50) and does not exceed it — that is the goal ("no
longer worse on article, still better on dialogue").

### 3. Training

QLoRA on a single 24GB GPU: 4-bit base + a LoRA adapter (rank 16-64, alpha ~2×rank,
lr ~1-2e-4, 1-3 epochs, max_seq_len ~1024, train-on-response-only). ~1-3 GPU-hours, ~$1-3.
Tooling: **Unsloth** or **LLaMA-Factory** (config-driven) or TRL `SFTTrainer` (lower level).

### 4. Eval and forgetting check

Evaluate on **MiLiC-Eval, both domains**, after SFT. Success = article ↑ toward NLLB **and**
dialogue held ≥ NLLB. If dialogue drops, the NLLB data was over-weighted — rebalance.

### 5. Back into the normal pipeline

Merge the LoRA adapter → standalone Gemma translator. Then it is *just a teacher*: point
`distill_data.py` at it (vLLM, instead of NLLB via CT2) and distill into the base-memory
slimt student exactly like NLLB. A properly SFT'd model emits only the translation and
stops, so no script-filtering gymnastics are needed for the KD data.

## Why NLLB-KD here, instead of NLLB's own training pairs?

The question: if we want article-domain parallel data, why generate it with NLLB rather than
feed the human/mined pairs that NLLB itself was trained on?

- **NLLB's *output* is cleaner than its *input*.** NLLB's per-pair Uyghur training data is
  sparse, mined (CCMatrix/CCAligned), and semantically misaligned. NLLB's value is the
  cross-lingual generalization baked into its 200-language *weights* — not recoverable from
  any single noisy bitext it saw. Its regenerated translations are more coherent and
  consistent than the raw pairs.
- **KD guarantees alignment.** NLLB regenerates the target from the source, so every KD pair
  is a real translation of its source. Raw mined pairs carry misalignment (pairs that look
  parallel but are not) that would directly teach wrong mappings. (Same reason KD is immune
  to source-corpus misalignment in `NOTES.md`.)
- **KD gives domain control + volume.** We choose the source text (article-domain EN-wiki,
  Uyghur MC²), so we generate pairs *in the domain we want* at whatever volume we want. The
  raw NLLB training set is a fixed, mostly-formal-web mix we cannot steer, and reconstructing
  it is impractical.
- **Single-model target style is easier to learn** (classic sequence-level KD): the student
  fits one coherent output mode rather than the heterogeneous union of many corpora.

**The exception — do NOT launder human pairs through NLLB.** For the *human-signal* portion
(clean human bitext: the community's pairs, Tatoeba/TED, and the dialogue set), feed the
**original human pairs directly**. Human references beat teacher output (the tl→en finetune
lesson in `NOTES.md`: human data lifts quality, more teacher data does not). Re-translating
human pairs through NLLB would throw away exactly the signal that matters. So: NLLB-KD for
the article *volume/domain* fill; original human pairs for the *quality/dialogue* signal.

## Risks / open questions

- **Forgetting the dialogue edge** under NLLB-heavy SFT. Mitigate: balance domains, LoRA (not
  full FT), modest epochs, eval both domains.
- **Semantic hallucination** in the few-shot CPT base (e.g. it invented "Friday, August 30"
  in one sample). Script-filtering (Arabic vs Latin) catches malformed/wrong-script output
  but NOT fluent mistranslations — SFT is what fixes those. Measure the semantic error rate
  on ~50 dialogue lines before trusting the base as anything more than a starting point.
- **Article ceiling** — WAS assumed = NLLB (~43). The 2026-07-15 gate lifts it: frontier
  generators hit 46-50 article en→ug (see Empirical validation), so a frontier-generated
  article slice pushes the teacher above NLLB. The remaining cap is distillation (~5-6 chrF
  student-vs-teacher into-target), landing the shipped en→ug student ~43-45.
- **Gemma license vs Apache**: pkupie is Gemma-3 (Gemma Terms + PUP, clashes with GPLv3);
  a clean version means CPT on Gemma-4 (Apache) first.

## Quick test first (gate before scaling)

1. LoRA-SFT pkupie on ~15-20k balanced mix (NLLB-generated article + whatever dialogue human
   data is on hand).
2. Re-run the MiLiC gate.
3. Read: article climbed toward NLLB **and** dialogue stayed ahead? → recipe works.
4. If yes → run the same thing with the community's real dialogue pairs. Same operation,
   better data. If dialogue collapsed → rebalance and retry before committing.

Cost ~$2, a couple GPU-hours. This directly tests the one open risk (does covering article
via NLLB cost the dialogue win) before any larger investment.

## Empirical validation (2026-07-15): frontier gate, SFT scale, generation tooling

Ran the frontier-vs-NLLB gate on MiLiC **en→ug** (the blocked direction), every model
scored on the **same 100 sentences** per domain (chrF++). Only en→ug is self-measured:
ug→en's target is English, which the frontier models saw as the translation *source*, so
it is contamination-excluded. The 100-sentence subset reproduces the full-set NLLB numbers
(NLLB-600M dialogue 34.9 here vs 33.6 full-set), so it is representative. Harness:
`milic_gate.py`; per-model outputs scored with sacrebleu chrF++ word_order=2.

| model      | dialogue | article |
|------------|----------|---------|
| Haiku 4.5  | 23.5     | 9.1     |
| NLLB-600M  | 34.9     | 32.4    |
| NLLB-1.3B  | 36.7     | 41.5    |
| Sonnet 5   | 40.5     | 45.9    |
| Opus 4.8   | 41.2     | 50.0    |
| Fable 5    | 45.3     | 49.0    |

- **Frontier beats NLLB-1.3B by +4 to +9 on both domains** → lifts the "article ceiling =
  NLLB" assumption: a frontier-generated article slice pushes the teacher to ~46-50 article.
- **Haiku is unusable for Uyghur** (below NLLB — genuine script but broken grammar/words).
  Low-resource ⇒ only large models carry the language; no cheap generator exists.
- **Domain split among the big three**: Fable best dialogue (45.3), Opus best article
  (50.0), within ~1 pt on the other's turf. Sonnet ~4 back on both but well above NLLB.
- On-device reality: the slimt student loses ~5-6 chrF distilling into-target, so a ~48-50
  teacher → shipped en→ug student ~43-45. Above NLLB, short of a formal-text "50".

### SFT scale — why ~10-15k, and why more hurts
The pairs are the model's **first and only translation training**: CPT taught the language
from *monolingual* Uyghur (zero pairs), Gemma base saw no translation task. SFT teaches the
*task* (align two capabilities the model already has), which saturates fast on a capable
base — proof: the raw CPT base with zero pairs already beats NLLB on dialogue few-shot.
Expected knee **~10-15k**; usable **gate at ~5k**; beyond ~30k the model just imitates the
generator and erodes the CPT dialogue edge. **Methodology is incremental**: grow the pool,
re-SFT *from the CPT base* each round (not continue-train the adapter), stop when MiLiC
plateaus. Every pair pools forward — no tokens wasted.

### Generation tooling — `scripts/opus-trainer/gen_sft.py`
Resumable translator via `claude -p` (**no API key**; runs on the Max plan). Lean flags
`--allowedTools '' --exclude-dynamic-system-prompt-sections` cut per-call harness overhead
19k→6.5k tokens; adaptive thinking adds ~nothing to translation, and `claude -p` quality
matches the API path (Sonnet dialogue 41.5 vs 40.1 on the same sentences). One file per
batch, rerun skips completed batches, assembles `pairs.jsonl` ({src,tgt}). Use
`--batch ~100-150` (bigger amortizes overhead but one malformed array loses the whole
batch). Cost-equiv (Max usage, not billed): **Opus en→ug article ~$0.006/pair measured @batch100**
(long wiki sentences; short/dialogue text runs cheaper). The first 2500 en→ug article
tranche cost ~$15 equiv, 0 batch failures, mean 98.8% Uyghur-script. Source pools via
`fetch_en_article.py` (streams English Wikipedia → clean sentence list; tighten its filter
to drop bibliography/citation lines — ~0.5% leaked through as low-value title-heavy pairs).

### Refined data recipe — generation collapses to article
Dialogue is **human pairs, used directly** (both directions, never laundered through a
model — the "don't launder human pairs" rule). So generation is only the article domain:
- **en→ug article** (hard, into-Uyghur) — **Opus** (source: English wiki/news). The
  high-value run; the gap NLLB can't fill. First tranche: 2500 pairs.
- **ug→en article** (easy, into-English) — Sonnet or NLLB, *not* Opus (source: Uyghur
  MC²/Wikipedia). NLLB-on-vast is disproportionate for a few-k pairs (KD-box lesson).
- First gate can be just en→ug-article(Opus) + human dialogue.

Only the article slice is paid (dialogue is human/free): 2500 article ≈ $15 equiv; a
5-7k article set ≈ $30-45. **First gate assembled 2026-07-15**: 2500 en→ug article
(`sft/article_en2ug/pairs.jsonl`, Opus) + human dialogue; ug→en article skipped until the
eval shows an into-English article weakness.

### First gate RESULT (2026-07-15) — mechanism works; single-direction SFT is catastrophic
LoRA-SFT'd CPT-Gemma-4B on the 2500 en→ug article pairs **only** (Unsloth QLoRA r32,
3 epochs, response-masked; vast 4090, ~$0.30). MiLiC first-100, chrF++, base 5-shot vs
SFT zero-shot:

| dir / domain      | base 5-shot | SFT  | Δ     |
|-------------------|-------------|------|-------|
| article  en→ug    | 37.65       | 41.58| +3.9  |
| dialogue en→ug    | 36.56       | 37.52| +1.0  |
| article  ug→en    | 45.44       | 1.33 | −44   |
| dialogue ug→en    | 47.54       | 0.86 | −47   |

Two findings: (1) **the recipe lifts the trained direction** on both domains (en→ug +3.9
article to NLLB-parity, +1.0 dialogue) from just 2500 pairs, and zero-shot (no few-shot
needed). (2) **Single-direction SFT catastrophically forgets the other direction** —
into-English collapsed 45→1 ("always output Uyghur" mode). ⇒ the full set MUST be
**balanced across both directions AND both domains** — this makes "both directions in one
training set" empirically mandatory, not optional. en→ug only reaching NLLB-parity at 2500
is expected below the ~10-15k knee (and a 4B student can't fully absorb Opus-50 data from
2500 examples); scale + balance next.

Training-stack notes (bleeding-edge TRL 0.24 / transformers 5.5 on the box): `SFTConfig`
renamed `max_seq_length`→`max_length`; `SFTTrainer` wants `processing_class` not
`tokenizer`; `SFTTrainer._prepare_dataset` map fails to pickle a torch `ConfigModuleInstance`.
Working path = **plain HF `Trainer` + manual response-masking**, and use `tok.tokenizer`
(the multimodal Gemma3 loads as a `Gemma3Processor` with no `.pad`). Scripts in scratchpad
`train_sft2.py` / `eval_sft.py` (not committed).

### COMET22 cross-check (2026-07-15) — chrF is NOT garbage-level; dialogue premise deflates
Ran Unbabel/wmt22-comet-da on the same en→ug 100-subsets (bigserver CPU). chrF vs COMET:

| model      | article chrF | article COMET | dialogue chrF | dialogue COMET |
|------------|-------------|---------------|---------------|----------------|
| Haiku      | 9.1  | 0.429 | 23.5 | 0.737 |
| NLLB-600M  | 32.5 | 0.680 | 34.9 | 0.885 |
| NLLB-1.3B  | 41.6 | 0.825 | 36.7 | 0.884 |
| Sonnet     | 45.9 | 0.874 | 40.5 | 0.892 |
| Opus       | 50.0 | 0.876 | 41.2 | 0.893 |
| Fable      | 49.0 | 0.877 | 45.3 | 0.900 |

Findings: (1) **~40 chrF Uyghur is NOT garbage** — NLLB-1.3B article 41.6 chrF = 0.825 COMET
(decent), dialogue 36.7 chrF = 0.884 (good). Confirms chrF absolute is the wrong yardstick
vs Mozilla's Latin-target packs; only Haiku is truly bad on both metrics. (2) **The
frontier-vs-NLLB gap is REAL on article** (0.825→0.876 COMET, +0.05) **but nearly ILLUSORY
on dialogue** (0.884→0.900, +0.015) — the +8 chrF frontier "win" on dialogue is mostly
surface paraphrase, which COMET sees through. This **weakens the doc's headline premise**
("dialogue is where specialization beats NLLB"). (3) On dialogue COMET barely discriminates
(all non-Haiku 0.88-0.90) — partly real (short colloquial = many valid forms), partly a
COMET limitation (XLM-R's Uyghur is low-resource → coarse, may under-discriminate).
**Caveat: neither metric is gold for Uyghur** (chrF over-penalizes morphology/paraphrase;
COMET may under-discriminate) — a ~30-50-line native-speaker check is the tiebreaker.
Implication: judge the teacher by **COMET, target ~0.87 article** (clearly above NLLB 0.825);
the SFT gate's chrF 41.6 ≈ NLLB-level ≈ ~0.825 COMET, so it must climb ~0.05 COMET on
article to justify distillation. Report COMET (not just chrF) on the next run. Scorer:
scratchpad `comet_score.py`.

### COMET ug→en (reverse) — CORRECTS the dialogue deflation; the spend map (2026-07-15)
Measured the reverse direction (Sonnet ug→en via `claude -p`, clean — fresh gen sees only
Uyghur source; NLLB from bigserver; COMET22). Full 2×2 NLLB→Sonnet:

| slice          | NLLB COMET      | Sonnet COMET | Δ      |
|----------------|-----------------|--------------|--------|
| en→ug article  | 0.825 (1.3b)    | 0.874        | +0.049 |
| en→ug dialogue | 0.884 (1.3b)    | 0.892        | +0.008 |
| ug→en article  | 0.851 (1.3b)    | 0.879        | +0.028 |
| ug→en dialogue | 0.783 (600m)    | 0.862        | **+0.079** |

**This corrects the earlier "dialogue premise deflates" call** — that was en→ug ONLY, where
COMET under-discriminates into low-resource Uyghur. On the RELIABLE into-English direction,
**ug→en dialogue is the BIGGEST Sonnet gap (+0.079)**: NLLB ug→en dialogue is genuinely weak
(0.783, *below* Mozilla's ~0.83 low-resource floor — az-en 0.831, be-en 0.792) and Sonnet
lifts it to 0.862. So dialogue IS a real teacher win; the en→ug-dialogue "flat" (+0.008) is a
COMET-into-Uyghur measurement artifact, not a true null. Lesson: **trust into-English COMET;
treat into-Uyghur COMET as a floor.**

**Sonnet `claude -p` spend map (by delta, biggest first):** (1) **ug→en dialogue +0.079** —
top priority, fixes NLLB's real hole; (2) **en→ug article +0.049** — clear; (3) ug→en article
+0.028 — modest; (4) en→ug dialogue +0.008 measured but uncertain (likely larger, COMET can't
see it). Generate slices 1-2 first. Mozilla reference class (their shipped low-resource
students, COMET22): az 0.831-0.837, fa 0.846, bn 0.847, ar 0.860, be-en 0.792 — so Uyghur at
~0.85-0.88 would be squarely shippable.

## Data / licensing summary

| asset | what | license | role |
|---|---|---|---|
| MC² (pkupie) | ~736 MB monolingual Uyghur (~150M tok) | CC0 | CPT data; ug→en article KD source |
| MiLiC-Eval (pkupie) | human en↔ug, article+dialogue, chrF++ | gated (access granted) | the eval |
| pkupie/gemma-3-4b-ug-cpt | Gemma-3-4B CPT'd on MC² | Gemma Terms + PUP | base (or SFT start) |
| Gemma-4 base | stronger base | Apache 2.0 | clean-license CPT target |
| NLLB-200-1.3B | teacher | CC-BY-NC | article KD generator (NC is fine — app is NC) |
| community pairs | human en↔ug, esp. dialogue | — | the real signal; SFT + finetune |

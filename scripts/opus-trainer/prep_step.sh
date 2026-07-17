#!/bin/bash
# pipe step wrapper for prep_data.py: full corpus prep in the prep image,
# emitting the 2-col pool (src \t tgt) the KD flow consumes. The workdir with
# raw zips and intermediates is deleted after the paste — the pool is the
# artifact, everything else is re-derivable.
#
# JOBS is explicit because nproc inside a --cpus-limited container reports the
# host's cores, not the quota (the vast-perf-traps lesson).
#
# VOCAB_MODE picks whether this prep trains the pair's joint SPM:
#   reuse — an existing joint vocab is seeded into the run (a re-prep of a pair
#           that already shipped). Emits pool.tsv only.
#   train — a NEW pair has no vocab; train the joint 32k unigram on both sides
#           and emit it as vocab.spm alongside the pool. A joint SPM is
#           pair-specific, so it cannot be borrowed from another language.
# The vocab is emitted as `.spm` because marian detects SentencePiece by
# extension — a `.model` name fails with "DefaultVocabulary must not contain
# empty lines" (NOTES pillar 4 gotcha (a)).
#
# Usage: prep_step.sh TGT_LANG JOBS OUT_DIR [SRC=en] [VOCAB_MODE=reuse] [ONLY=]
set -euo pipefail

TGT=$1; JOBS=$2; OUT=$3; SRC=${4:-en}; VOCAB_MODE=${5:-reuse}; ONLY=${6:-}
PAIR="${SRC}${TGT}"

case "$VOCAB_MODE" in
  reuse) SPM_ARGS=(--skip-spm) ;;
  train) SPM_ARGS=() ;;
  *) echo "VOCAB_MODE must be 'reuse' or 'train', got '$VOCAB_MODE'" >&2; exit 2 ;;
esac

ONLY_ARGS=()
[ -n "$ONLY" ] && ONLY_ARGS=(--only "$ONLY")

python3 /scripts/prep_data.py --lang "$TGT" --src "$SRC" --workdir "$OUT/work" \
  --jobs "$JOBS" "${SPM_ARGS[@]}" "${ONLY_ARGS[@]}" --lid-model /opt/lid.176.ftz

paste <(zcat "$OUT/work/clean/train.${PAIR}.${SRC}.gz") \
      <(zcat "$OUT/work/clean/train.${PAIR}.${TGT}.gz") > "$OUT/pool.tsv"

if [ "$VOCAB_MODE" = train ]; then
  cp "$OUT/work/spm/vocab.${PAIR}.model" "$OUT/vocab.spm"
fi

rm -rf "$OUT/work"
wc -l "$OUT/pool.tsv"

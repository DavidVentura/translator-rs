#!/bin/bash
# Phase 3b: quantize-aware finetune of the KD-trained student on CLEAN parallel,
# behind the same OpusTrainer input augmentation as train_student.sh (case/typos/
# noise/end-punct) so the final polish is also robust to messy/casual input.
#
# The KD-only student (trained on teacher forward-translations of mined web text)
# underperforms on clean-domain test sets. Finetuning on clean parallel closes
# the gap: en->tl went to 57.9 chrF++ vs 59.4 teacher (KD-only was much lower via
# the runtime). This is NOT optional for a shippable model.
#
# OpusTrainer pipes the augmented 3-col TSV into marian's stdin, so there is NO
# --train-sets; --shuffle batches + --no-restore-corpus are required for the
# non-seekable stream.
#
# student.finetune.yml sets quantize-bits: 8 which REQUIRES --sync-sgd.
# VALID_TSV must be a 2-COLUMN tsv (src\ttrg) -- marian validation wants 2 fields,
# a 3-col valid (with alignment) aborts with "Excessive field(s)".
#
# Needs `pip install opustrainer` (console script opustrainer-train) in the env.
# MARIAN env var = path to the marian trainer binary (default /root/marian-dev/build/marian).
#
# Usage: finetune_student.sh CLEAN_TSV VOCAB PRETRAINED_NPZ VALID_2COL_TSV MODEL_OUT [DEVICES=0]
set -euo pipefail

TSV=$1; VOCAB=$2; PRE=$3; VALID=$4; OUT=$5; DEVICES=${6:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"
MARIAN=${MARIAN:-$(command -v marian || echo /root/marian-dev/build/marian)}
mkdir -p "$(dirname "$OUT")"
if [[ "$VOCAB" != *.spm ]]; then
  ln -sf "$VOCAB" "$(dirname "$OUT")/vocab.spm" && VOCAB="$(dirname "$OUT")/vocab.spm"
fi
if [[ "$PRE" != *.npz && "$PRE" != *.bin ]]; then
  ln -sf "$PRE" "$(dirname "$OUT")/pretrained.npz" && PRE="$(dirname "$OUT")/pretrained.npz"
fi

OT_CFG="$(dirname "$OUT")/config.opustrainer.finetune.yml"
sed "s|__TSV__|$TSV|g" "$HERE/configs/opustrainer.student.yml" > "$OT_CFG"

opustrainer-train --config "$OT_CFG" --log-level INFO \
  "$MARIAN" \
  -c "$HERE/configs/student.base-memory.yml" "$HERE/configs/student.finetune.yml" \
  --tsv --tsv-fields 3 \
  --shuffle batches --no-restore-corpus \
  --vocabs "$VOCAB" "$VOCAB" --dim-vocabs 32000 32000 \
  --pretrained-model "$PRE" --model "$OUT" \
  --valid-sets "$VALID" --valid-metrics ce-mean-words \
  --sync-sgd --devices $DEVICES --workspace 9000 --keep-best --overwrite

echo "done: ${OUT%.npz}.best-ce-mean-words.npz (use the .best for quantization)"

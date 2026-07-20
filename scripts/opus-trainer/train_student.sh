#!/bin/bash
# Phase 3: train a base-memory SSRU student with guided-alignment (KD phase),
# behind OpusTrainer input augmentation (case/typos/noise/end-punct) for
# robustness to messy, casual, and OOD input. Run inside a marian image (CPU:
# marian-*:cpu; GPU: a CUDA marian build, see docker/Dockerfile.marian-cuda).
# Follow with finetune_student.sh on clean data.
#
# OpusTrainer reads the 3-col TSV, perturbs a fraction of lines in place (same
# line count), and pipes the augmented stream into marian's stdin. Marian reads
# --tsv from stdin, so there is NO --train-sets; --shuffle batches +
# --no-restore-corpus are required because the corpus is a non-seekable stream.
#
# TRAIN_TSV is a 3-col TSV: src \t trg \t pharaoh-alignments (from align.sh).
# VALID_TSV must be a 2-COLUMN tsv (src\ttrg) -- marian validation wants 2 fields;
# a 3-col valid aborts with "Excessive field(s)".
# DEVICES e.g. "0" / "0 1" for GPU; omit for CPU.
# MARIAN env var overrides the binary path (default /root/marian-dev/build/marian).
#
# Needs `pip install opustrainer` (console script opustrainer-train) in the env.
# Gotcha: the trainer target is `marian_train` (underscored); `make marian` builds
# only the static lib. See docker/Dockerfile.marian-cuda.
#
# Usage: train_student.sh TRAIN_TSV VOCAB_SPM VALID_2COL_TSV MODEL_OUT_NPZ [DEVICES]
set -euo pipefail
[ -d /work/out ] && echo train > /work/out/.phase || true

TSV=$1; VOCAB=$2; VALID=$3; OUT=$4; DEVICES=${5:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
MARIAN=${MARIAN:-$(command -v marian || echo /root/marian-dev/build/marian)}
mkdir -p "$(dirname "$OUT")"
# marian detects SentencePiece vocabs by the .spm extension; pipe materializes
# inputs under bare names, so re-expose the vocab with its extension.
if [[ "$VOCAB" != *.spm ]]; then
  ln -sf "$VOCAB" "$(dirname "$OUT")/vocab.spm" && VOCAB="$(dirname "$OUT")/vocab.spm"
fi

if [ -n "$DEVICES" ]; then
  DEV=(--devices $DEVICES --workspace 9000)
else
  DEV=(--cpu-threads "$(nproc)" --workspace 4000)
fi

OT_CFG="$(dirname "$OUT")/config.opustrainer.yml"
sed "s|__TSV__|$TSV|g" "$HERE/configs/opustrainer.student.yml" > "$OT_CFG"

opustrainer-train --config "$OT_CFG" --log-level INFO \
  "$MARIAN" \
  -c "$HERE/configs/student.base-memory.yml" "$HERE/configs/student.train.yml" \
  --tsv --tsv-fields 3 \
  --shuffle batches --no-restore-corpus \
  --vocabs "$VOCAB" "$VOCAB" --dim-vocabs 32000 32000 \
  --model "$OUT" \
  --valid-sets "$VALID" --valid-metrics ce-mean-words \
  --keep-best --overwrite \
  "${DEV[@]}"

echo "done: ${OUT%.npz}.best-ce-mean-words.npz"

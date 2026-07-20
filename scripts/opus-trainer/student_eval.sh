#!/bin/bash
# pipe step wrapper: bench the packed student on FLORES AND the check set.
# lang/src arrive as ARGS, not env: the job protocol passes an args file and a
# script path, never environment.
#
# Usage: student_eval.sh LANG SRC MODEL VOCAB SHORTLIST C_SRC C_REF OUT_HYP OUT_METRICS OUT_REVIEW
set -euo pipefail
LANG_=$1; SRC=$2; MODEL=$3; VOCAB=$4; SHORTLIST=$5; C_SRC=$6; C_REF=$7
OUT_HYP=$8; OUT_M=$9; OUT_R=${10}

python3 /scripts/benchmark_slimt.py --lang "$LANG_" --src "$SRC" \
    --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
    --probes "$C_SRC" --probe-out "$OUT_HYP" | tee "$OUT_M"

python3 /scripts/probe_review.py "$C_SRC" "$OUT_R" --hyp student "$OUT_HYP"

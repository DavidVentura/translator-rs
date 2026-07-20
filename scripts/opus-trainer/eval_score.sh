#!/bin/bash
# pipe step wrapper: score FLORES + probes, render the review artifact.
# Emits numbers and a readable side-by-side; never a pass/fail verdict.
#
# Usage: eval_score.sh F_HYP F_REF F_SRC P_HYP P_SRC P_REF OUT_METRICS OUT_REVIEW
set -euo pipefail
F_HYP=$1; F_REF=$2; F_SRC=$3; P_HYP=$4; P_SRC=$5; P_REF=$6; OUT_M=$7; OUT_R=$8

python3 /scripts/eval_pair.py \
    --flores-hyp "$F_HYP" --flores-ref "$F_REF" --flores-src "$F_SRC" \
    --probe-hyp "$P_HYP" --probe-src "$P_SRC" --probe-ref "$P_REF" \
    --out "$OUT_M"

python3 /scripts/probe_review.py "$P_SRC" "$OUT_R" --hyp system "$P_HYP"

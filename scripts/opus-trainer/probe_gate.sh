#!/bin/bash
# pipe step wrapper: decode BOTH probe directions with Hy-MT2 on one rented box.
#
# Usage: probe_gate.sh MODEL EN_SRC TL_SRC OUT_DIR
set -euo pipefail

MODEL=$1; EN_SRC=$2; TL_SRC=$3; OUT_DIR=$4

mkdir -p "$OUT_DIR"
python3 /scripts/probe_decode.py "$MODEL" "$EN_SRC" "Filipino" "$OUT_DIR/en2tl.hyp"
python3 /scripts/probe_decode.py "$MODEL" "$TL_SRC" "English"  "$OUT_DIR/tl2en.hyp"

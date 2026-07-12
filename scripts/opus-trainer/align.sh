#!/bin/bash
# Phase 3 prep: word alignment (fast_align) -> 3-col guided-alignment TSV for training.
# Run inside a marian image with fast_align built under $TOOLS (clab/fast_align).
#
# Alignments for GUIDED-ALIGNMENT TRAINING are WHITESPACE-level (marian maps
# word->subword internally). The lexical SHORTLIST is a separate thing and must
# be SPM-subword level -- see shortlist.sh. Don't reuse this TSV's alignment for
# the shortlist.
#
# tr '\t'->' ' guards against stray tabs in a field (marian splitTsv aborts on
# any line with >2 tabs).
#
# Usage: align.sh SRC_FILE TRG_FILE OUT_DIR [TOOLS_DIR=/work]
#   -> OUT_DIR/train.tsv  (src \t trg \t pharaoh-align)  for marian --tsv --tsv-fields 3
set -euo pipefail

SRC=$1; TRG=$2; OUT=$3; TOOLS=${4:-/work}
FA="$TOOLS/fast_align/build"
mkdir -p "$OUT"

paste <(tr '\t' ' ' < "$SRC") <(tr '\t' ' ' < "$TRG") | sed 's/\t/ ||| /' > "$OUT/fa_in"
"$FA/fast_align" -i "$OUT/fa_in" -d -o -v      > "$OUT/fwd" 2>/dev/null
"$FA/fast_align" -i "$OUT/fa_in" -d -o -v -r   > "$OUT/rev" 2>/dev/null
"$FA/atools" -i "$OUT/fwd" -j "$OUT/rev" -c grow-diag-final-and > "$OUT/sym"

paste <(tr '\t' ' ' < "$SRC") <(tr '\t' ' ' < "$TRG") "$OUT/sym" > "$OUT/train.tsv"
echo "done: $OUT/train.tsv (3-col guided-align, whitespace-level)"

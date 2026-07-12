#!/bin/bash
# Phase 4b: build the lexical shortlist for a slimt pack.
#
# CRITICAL: the shortlist must be over SPM SUBWORDS, not raw words. slimt's
# shortlisted output projection is over the SPM vocab, so a word-level shortlist
# constrains it to the wrong candidates -> garbage / repetition loops (this cost
# a long debug: model was fine at 58 chrF++, a word-level shortlist dragged the
# slimt score to 34). Mirrors mozilla pipeline/alignments/generate-shortlist.sh.
#
# Run inside the marian-bmt image (spm_encode, fast_align, extract_lex, marian-conv).
#
# Usage: shortlist.sh SRC_FILE TRG_FILE VOCAB_SPM OUT_DIR [TOOLS=/work]
#   -> OUT_DIR/lex.50.50.s2t.bin   (binarized; goes in the pack next to the model)
set -euo pipefail

SRC=$1; TRG=$2; VOCAB=$3; OUT=$4; TOOLS=${5:-/work}
BMT=/opt/marian-dev/build
FA="$TOOLS/fast_align/build"; XL="$TOOLS/extract-lex/build"
mkdir -p "$OUT"

# 1) SPM-segment both sides
"$BMT/spm_encode" --model "$VOCAB" < "$SRC" > "$OUT/spm.src"
"$BMT/spm_encode" --model "$VOCAB" < "$TRG" > "$OUT/spm.trg"

# 2) align the SUBWORD streams
paste "$OUT/spm.src" "$OUT/spm.trg" | sed 's/\t/ ||| /' > "$OUT/fa_in"
"$FA/fast_align" -i "$OUT/fa_in" -d -o -v    > "$OUT/fwd" 2>/dev/null
"$FA/fast_align" -i "$OUT/fa_in" -d -o -v -r > "$OUT/rev" 2>/dev/null
"$FA/atools" -i "$OUT/fwd" -j "$OUT/rev" -c grow-diag-final-and > "$OUT/aln"

# 3) lexical table (extract_lex args are TARGET then SOURCE); drop NULL alignments
"$XL/extract_lex" "$OUT/spm.trg" "$OUT/spm.src" "$OUT/aln" "$OUT/lex.s2t" "$OUT/lex.t2s"
grep -v NULL "$OUT/lex.s2t" > "$OUT/lex.s2t.clean"

# 4) binarize (firstNum bestNum threshold = 50 50 0)
"$BMT/marian-conv" --shortlist "$OUT/lex.s2t.clean" 50 50 0 \
  --dump "$OUT/lex.50.50.s2t.bin" --vocabs "$VOCAB" "$VOCAB"

echo "done: $OUT/lex.50.50.s2t.bin"

#!/bin/bash
# KD source + per-line references from the cleaned 2-col pool (src \t tgt).
# The old recipe deduped the source SIDE alone (sort -u on one column), which
# threw away the pairing — but extract-best needs each KD line's own bitext
# target as its reference, so dedup by the KD-source column and keep the pair.
#
# KD_COL is the column that becomes the KD source (2 = the pool's tgt side, e.g.
# sw in an en\tsw pool feeding a sw->en student); the other column is the ref.
#
# Usage: build_kd_source.sh POOL_TSV KD_COL OUT_DIR
set -euo pipefail

POOL=$1; KD_COL=$2; OUT=$3
REF_COL=$(( KD_COL == 1 ? 2 : 1 ))

# Both fields must be non-blank: "text\t" still has NF==2, and a zero-token side
# downstream sends fast_align's EM to nan (the align_ensw defect class).
awk -F'\t' -v k="$KD_COL" \
  'NF == 2 && $1 ~ /[^[:space:]]/ && $2 ~ /[^[:space:]]/ && !seen[$k]++' "$POOL" \
  | shuf --random-source=<(yes 42) > "$OUT/pairs"
cut -f"$KD_COL"  "$OUT/pairs" > "$OUT/kd_src"
cut -f"$REF_COL" "$OUT/pairs" > "$OUT/kd_ref"
rm "$OUT/pairs"
wc -l "$OUT/kd_src" "$OUT/kd_ref"

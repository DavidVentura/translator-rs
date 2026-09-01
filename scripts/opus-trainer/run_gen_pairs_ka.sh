#!/usr/bin/env bash
# Drive the en<->ka sign/label/menu/UI pair generation round by round.
#
# Rounds are separate invocations on purpose: `have` (what the set already
# holds, named in every prompt to suppress repeats) is read from the job files
# at process start, so a round that runs inside the same process as the round
# before it cannot see what that round wrote. One invocation per round makes
# each round's prompts aware of every earlier round.
#
#   nohup ./run_gen_pairs_ka.sh 4 > data/gen_pairs/ka/gen.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")"
ROUNDS="${1:-3}"
OUT=data/gen_pairs/ka
PY=venv/bin/python

mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  echo "=== round $r/$ROUNDS  $(date -Is)"
  "$PY" gen_pairs.py gen \
    --spec configs/gen_pairs.ka.json \
    --out "$OUT" \
    --rounds "$r" \
    --workers 4 \
    --shuffle-seed 11
done
echo "=== generation done $(date -Is)"

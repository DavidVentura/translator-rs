#!/usr/bin/env python3
"""Pick the KD training target from teacher n-best lists (mozilla extract-best).

Per line, keep the hypothesis with the best sentence chrF against the human
reference: the target stays teacher-fluent (easy for the student to compress)
while the reference acts as a judge pulling the choice toward human phrasing.
Beam rank-1 is only "best" under the teacher's own distribution — its favorite
errors included; a lower-ranked hypothesis is often closer to the reference.

Selection is GATED by bicleaner score: below the threshold the reference may be
semantically misaligned, and "closest to reference" would mean "closest to
noise", so those lines keep rank-1 (which is exactly the old 1-best pipeline).
Without --gate-scores every line is selected against its reference.

Pure CPU and embarrassingly parallel: run per shard on the KD box right after
distill_data.py --nbest, while the box is still up.

    pip install sacrebleu
    python extract_best.py --nbest shard.nbest.tsv.gz --ref shard.ref.gz \
        --gate-scores shard.bic --out shard.sel.gz --jobs 16
"""

import argparse
import gzip
import sys
from itertools import islice, zip_longest
from multiprocessing import Pool

from sacrebleu.metrics import CHRF

_CHRF = None


def opener(p: str):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def writer(p: str):
    return gzip.open(p, "wt", encoding="utf-8") if p.endswith(".gz") else open(p, "w", encoding="utf-8")


def _init_worker() -> None:
    global _CHRF
    _CHRF = CHRF()


def _pick_batch(rows: list[tuple[str, str, bool]]) -> tuple[list[str], int, int]:
    out = []
    selected = moved = 0
    for nbest_line, ref, gated in rows:
        hyps = nbest_line.split("\t")
        if gated and ref and len(hyps) > 1:
            scores = [_CHRF.sentence_score(h, [ref]).score for h in hyps]
            best = scores.index(max(scores))
            out.append(hyps[best])
            selected += 1
            moved += best != 0
        else:
            out.append(hyps[0])
    return out, selected, moved


def rows(args):
    gates = opener(args.gate_scores) if args.gate_scores else None
    with opener(args.nbest) as fn, opener(args.ref) as fr:
        for nb, ref in zip_longest(fn, fr):
            if nb is None or ref is None:
                sys.exit("line-count mismatch between --nbest and --ref")
            if gates is not None:
                g = gates.readline()
                if not g:
                    sys.exit("line-count mismatch between --nbest and --gate-scores")
                gated = float(g) >= args.gate_threshold
            else:
                gated = True
            yield nb.rstrip("\n"), ref.rstrip("\n"), gated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nbest", required=True, help="N-column TSV from distill_data.py --nbest (.gz ok)")
    ap.add_argument("--ref", required=True, help="human references, line-aligned with --nbest (.gz ok)")
    ap.add_argument("--out", required=True, help="selected one-target-per-line output (.gz ok)")
    ap.add_argument("--gate-scores", default="", help="per-line bicleaner scores; below threshold keep rank-1")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    it = rows(args)
    n = selected = moved = 0
    with writer(args.out) as fout, Pool(args.jobs, initializer=_init_worker) as pool:
        batches = iter(lambda: list(islice(it, args.batch)), [])
        for picked, sel, mov in pool.imap(_pick_batch, batches, chunksize=1):
            for line in picked:
                fout.write(line + "\n")
            n += len(picked)
            selected += sel
            moved += mov
            if n % 100000 < args.batch:
                print(f"  {n}", end="\r", file=sys.stderr)
    print(
        f"\nDONE {n} lines -> {args.out}: {selected} gated-in ({selected / max(n, 1):.1%}), "
        f"{moved} picked a non-rank-1 hypothesis ({moved / max(selected, 1):.1%} of gated)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

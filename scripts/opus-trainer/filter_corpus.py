#!/usr/bin/env python3
"""Drop web junk and byte-fallback garbage from a Uyghur corpus before it is decoded.

Two filters the round-1 pipeline had no equivalent of, both calibrated against the
2.46M certain-Uyghur bucket and the 1012-line FLORES devtest rather than by eye:

  JUNK  — Discuz! forum watermarks, the repeated forum footer, and Chinese-mixed
          rows. Each fires on 0 of 1012 FLORES lines. Counting punctuation instead
          does NOT discriminate (>=4 ascii punct hits 11.9% of HPLT but also 7.7%
          of FLORES gold and 7.5% of the round-1 corpus), so only precise forms.

  FERTILITY — SPM pieces per word against the joint vocab, a poor-man's CCNet
          perplexity that needs no new model: byte_fallback encodes text the vocab
          never learned at roughly one piece per BYTE, so mojibake, space-stripped
          text and decoration rules explode while clean prose does not. FLORES gold
          never exceeds 3.00; the default 4.0 clears its whole range with margin and
          costs 0.22% of the round-1 corpus that trained a working student (which is
          itself mojibake that should not have been in there).

Runs inside an image with sentencepiece (prep:next).

Usage:
    filter_corpus.py --vocab vocab.spm --in hplt.new --out hplt.src
    filter_corpus.py --vocab vocab.spm --in train.tsv --column 1 --out kept.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm

# Kept in sync with segment_mono.JUNK_RES; duplicated rather than imported because
# this runs inside the prep image where only this file is mounted.
JUNK_RES = (
    re.compile(r"[A-Za-z] ?[%&$#*;,)(]+ ?[A-Za-z0-9]"),
    re.compile(r"مەزمۇنلار پۈتۈنلەي"),
    re.compile(r"[一-鿿]"),
)


def fertility(sp: spm.SentencePieceProcessor, text: str) -> float:
    words = len(text.split())
    if not words:
        return 0.0
    return len(sp.encode(text)) / words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--column", type=int, default=None, help="1-based TSV field to judge; default whole line")
    ap.add_argument("--max-fertility", type=float, default=4.0)
    ap.add_argument("--dropped", type=Path, default=None)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.vocab))

    counts: Counter[str] = Counter()
    dropped = args.dropped.open("w", encoding="utf-8") if args.dropped else None
    with args.src.open(encoding="utf-8") as f, args.out.open("w", encoding="utf-8") as out:
        for line in f:
            row = line.rstrip("\n")
            counts["total"] += 1
            field = row.split("\t")[args.column - 1] if args.column else row
            if any(rx.search(field) for rx in JUNK_RES):
                counts["junk"] += 1
                if dropped:
                    dropped.write(row + "\n")
                continue
            if fertility(sp, field) > args.max_fertility:
                counts["fertility"] += 1
                if dropped:
                    dropped.write(row + "\n")
                continue
            counts["kept"] += 1
            out.write(row + "\n")
    if dropped:
        dropped.close()

    t = counts["total"]
    if t == 0:
        raise SystemExit("no input rows")
    for k in ("junk", "fertility", "kept"):
        print(f"  {k:10s} {counts[k]:>10,}  {100.0 * counts[k] / t:6.2f}%", file=sys.stderr)
    print(f"  {'total':10s} {t:>10,}", file=sys.stderr)


if __name__ == "__main__":
    main()

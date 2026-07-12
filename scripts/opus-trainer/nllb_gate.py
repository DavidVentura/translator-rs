#!/usr/bin/env python3
"""Gate NLLB-200 as a teacher on FLORES-200 devtest (chrF++/spBLEU).

Same idea as validate_teacher.py but for the languages OPUS-MT lacks/is weak on.
NLLB uses lang-code tokens (eng_Latn, swh_Latn, urd_Arab, ...). Reuses the FLORES
tarball cache. CPU is slow for the 600M model, so keep --limit small.

    ./venv/bin/python nllb_gate.py --pairs eng_Latn-swh_Latn,swh_Latn-eng_Latn --limit 100
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CACHE = Path(__file__).resolve().parent / ".cache"


def flores_devtest(code: str) -> list[str]:
    root = CACHE / "flores200_dataset"
    if not root.exists():
        CACHE.mkdir(exist_ok=True)
        tar = CACHE / "flores200_dataset.tar.gz"
        if not tar.exists():
            urllib.request.urlretrieve(FLORES_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(CACHE)
    return (root / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="comma list of FLORES pairs src_code-tgt_code")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()

    for pair in args.pairs.split(","):
        src_code, tgt_code = pair.split("-")
        src = flores_devtest(src_code)[: args.limit]
        ref = flores_devtest(tgt_code)[: args.limit]
        tok.src_lang = src_code
        bos = tok.convert_tokens_to_ids(tgt_code)
        hyps: list[str] = []
        with torch.no_grad():
            for i in range(0, len(src), args.batch):
                enc = tok(src[i : i + args.batch], return_tensors="pt", padding=True, truncation=True, max_length=256)
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=args.beam, max_length=256)
                hyps.extend(tok.batch_decode(gen, skip_special_tokens=True))
                print(f"  {pair}: {min(i + args.batch, len(src))}/{len(src)}", end="\r", file=sys.stderr)
        chrf = sacrebleu.corpus_chrf(hyps, [ref], word_order=2)
        sp = sacrebleu.corpus_bleu(hyps, [ref], tokenize="flores200")
        print(f"{src_code}->{tgt_code}: chrF++ {chrf.score:.2f}  spBLEU {sp.score:.2f}  (n={len(hyps)})")


if __name__ == "__main__":
    main()

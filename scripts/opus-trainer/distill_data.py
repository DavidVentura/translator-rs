#!/usr/bin/env python3
"""Phase 2: sequence-level KD data via CTranslate2 forward-translation.

Converts a teacher (OPUS-MT or NLLB-200) to CTranslate2 (int8) and translates a
source corpus, emitting synthetic targets aligned line-by-line with the source.
The (source, synthetic-target) pair is the student's KD training data. Meant to
run on a rented GPU box.

OPUS-MT teachers are self-describing (the tokenizer knows the pair). NLLB is a
single multilingual model, so pass --src-lang/--tgt-lang FLORES codes; that sets
the tokenizer source language and a target-language prefix token for the decoder.

    pip install ctranslate2 transformers sentencepiece sacremoses
    python distill_data.py --model Helsinki-NLP/opus-mt-en-tl \
        --src kd.en.gz --out kd.en2tl.tl.gz            # en->tl student targets (OPUS-MT)
    python distill_data.py --model facebook/nllb-200-distilled-600M \
        --src-lang tgl_Latn --tgt-lang eng_Latn \
        --src kd.tl.gz --out kd.tl2en.en.gz            # tl->en student targets (NLLB)
"""

import argparse
import gzip
import subprocess
import sys
from collections import deque
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer


def opener(p: str):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def writer(p: str):
    return gzip.open(p, "wt", encoding="utf-8") if p.endswith(".gz") else open(p, "w", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF teacher id (OPUS-MT or NLLB-200)")
    ap.add_argument("--src", required=True, help="source text (.gz ok)")
    ap.add_argument("--out", required=True, help="synthetic target out (.gz ok)")
    ap.add_argument("--src-lang", default="", help="NLLB source FLORES code (e.g. tgl_Latn); enables NLLB mode")
    ap.add_argument("--tgt-lang", default="", help="NLLB target FLORES code (e.g. eng_Latn)")
    ap.add_argument("--ct2-dir", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--inter-threads", type=int, default=2)
    ap.add_argument("--queue-depth", type=int, default=3, help="batches kept in flight to keep the GPU fed")
    args = ap.parse_args()

    nllb = bool(args.src_lang or args.tgt_lang)
    if nllb and not (args.src_lang and args.tgt_lang):
        ap.error("NLLB mode needs both --src-lang and --tgt-lang")

    ct2_dir = args.ct2_dir or "ct2_" + args.model.split("/")[-1].replace("-", "_")
    if not Path(ct2_dir).exists():
        print(f"converting {args.model} -> {ct2_dir} ({args.compute_type})", file=sys.stderr)
        subprocess.run(
            ["ct2-transformers-converter", "--model", args.model,
             "--output_dir", ct2_dir, "--quantization", args.compute_type],
            check=True,
        )
    tok = AutoTokenizer.from_pretrained(args.model)
    if nllb:
        tok.src_lang = args.src_lang
    target_prefix = [args.tgt_lang] if nllb else None
    translator = ctranslate2.Translator(
        ct2_dir, device=args.device, compute_type=args.compute_type, inter_threads=args.inter_threads,
    )

    n = 0
    # Submit each tokenized batch asynchronously so the GPU works on in-flight
    # batches while the next one is being tokenized/decoded on the CPU. Results
    # are drained in submission order, so output stays aligned with input.
    pending: deque = deque()

    def submit(lines: list[str]):
        # Cap source length to the model's max position encodings (512); an
        # over-long line otherwise crashes CT2 with a position-encoding error.
        toks = [tok.convert_ids_to_tokens(tok.encode(x, truncation=True, max_length=512)) for x in lines]
        kwargs = {"target_prefix": [target_prefix] * len(lines)} if target_prefix else {}
        return translator.translate_batch(
            toks, beam_size=args.beam, max_decoding_length=args.max_len, asynchronous=True, **kwargs,
        )

    def drain(fout) -> None:
        nonlocal n
        for r in pending.popleft():
            hyp = r.result().hypotheses[0]
            # NLLB decodes the target-language prefix token back out first; drop it.
            if target_prefix and hyp and hyp[0] == args.tgt_lang:
                hyp = hyp[1:]
            out = tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True)
            fout.write(out.replace("\n", " ") + "\n")
            n += 1
        print(f"  {n}", end="\r", file=sys.stderr)

    with opener(args.src) as fin, writer(args.out) as fout:
        batch: list[str] = []
        for line in fin:
            batch.append(line.rstrip("\n"))
            if len(batch) >= args.batch:
                pending.append(submit(batch))
                batch = []
                if len(pending) >= args.queue_depth:
                    drain(fout)
        if batch:
            pending.append(submit(batch))
        while pending:
            drain(fout)
    print(f"\nDONE {n} lines -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

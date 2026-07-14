#!/usr/bin/env python3
"""Score one hypothesis file against one reference file with the SAME metrics as
the teacher gates (chrF++ word_order=2, spBLEU flores200 tokenizer), so student
numbers line up directly against nllb_gate.py / hy_mt2_gate.py output.

    python chrf_score.py <label> <hyp> <ref>
"""

import sys

import sacrebleu


def main() -> None:
    label, hyp_path, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
    hyps = open(hyp_path, encoding="utf-8").read().splitlines()
    refs = open(ref_path, encoding="utf-8").read().splitlines()
    if len(hyps) != len(refs):
        sys.exit(f"{label}: line-count mismatch hyp={len(hyps)} ref={len(refs)}")
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200")
    print(f"{label}: chrF++ {chrf.score:.2f}  spBLEU {bleu.score:.2f}  (n={len(hyps)})")


if __name__ == "__main__":
    main()

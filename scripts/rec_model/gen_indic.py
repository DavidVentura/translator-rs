"""Synthetic Indic text-line strips for PP-OCRv6 rec fine-tuning — one MERGED
model over Bengali, Gujarati, Kannada, Malayalam (+ Latin for mixed content).

Indic reordering (pre-base matras, conjuncts) is LOCAL to a syllable, and HarfBuzz
keeps clusters in logical order, so we train on LOGICAL-order labels and let HB do
the visual layout (recgen Spec.reorder=True). CTC + the CNN receptive field absorb
the local glyph reorder; no downstream reordering at inference (unlike Hebrew RTL).
Verified via the HB cluster probe — see train_paddle_rec.md.

Each corpus line is a single script (concatenate the per-language Leipzig corpora);
a font covering the line's chars is auto-selected. ~25% Latin lines keep Latin alive.

  python gen_indic.py --out /tmp/indic --n 2000 --corpus indic_corpus.txt
  python gen_indic.py --out /tmp/indic-insp --n 32 --inspect --corpus indic_corpus.txt
"""

import random

import recgen

LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
PUNCT = " .,:;!?'\"()[]/%-+&@#।॥"  # incl. Indic danda + double danda (shared sentence terminators)

# Zero-width joiner/non-joiner steer conjunct formation but are invisible; strip them
# from text/labels (the model can't predict an invisible char) and accept canonical forms.
ZW = str.maketrans({"‌": None, "‍": None})

# Per-script Unicode blocks (for charset/coverage) and the assigned consonant /
# dependent-vowel (matra) / independent-vowel / digit subranges (for random fallback).
SCRIPTS = {
    "beng": {"lang": "bn", "block": (0x0980, 0x09FF), "cons": (0x0995, 0x09B9), "matra": (0x09BE, 0x09CC), "digit": (0x09E6, 0x09EF)},
    "gujr": {"lang": "gu", "block": (0x0A80, 0x0AFF), "cons": (0x0A95, 0x0AB9), "matra": (0x0ABE, 0x0ACC), "digit": (0x0AE6, 0x0AEF)},
    "knda": {"lang": "kn", "block": (0x0C80, 0x0CFF), "cons": (0x0C95, 0x0CB9), "matra": (0x0CBE, 0x0CCC), "digit": (0x0CE6, 0x0CEF)},
    "mlym": {"lang": "ml", "block": (0x0D00, 0x0D7F), "cons": (0x0D15, 0x0D39), "matra": (0x0D3E, 0x0D4C), "digit": (0x0D66, 0x0D6F)},
}

CHARSET = LATIN + DIGITS + PUNCT + "".join(
    chr(c) for s in SCRIPTS.values() for c in range(s["block"][0], s["block"][1] + 1)
)

# Map a codepoint to its script for routing corpus lines.
_BLOCK2SCRIPT = {name: s["block"] for name, s in SCRIPTS.items()}


def _line_script(line: str) -> str | None:
    for ch in line:
        for name, (lo, hi) in _BLOCK2SCRIPT.items():
            if lo <= ord(ch) <= hi:
                return name
    return None


def _rand_word(rng, s):
    out = []
    for _ in range(rng.randint(1, 5)):
        syl = chr(rng.randint(*s["cons"]))
        if rng.random() < 0.55:
            syl += chr(rng.randint(*s["matra"]))
        out.append(syl)
    return "".join(out)


def _latin_line(rng):
    toks = []
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.4:
            toks.append(str(rng.randint(0, 99999)))
        else:
            toks.append("".join(rng.choice(LATIN) for _ in range(rng.randint(2, 9))))
    return " ".join(toks)


def gen_pair(rng: random.Random, corpus: list[str]) -> tuple[str, str]:
    budget = rng.randint(6, recgen.MAX_LABEL_LEN)
    if rng.random() < 0.25:
        text = recgen.join_to_budget(rng, _latin_line(rng).split(), budget)
    elif corpus and rng.random() < 0.9:
        words = rng.choice(corpus).translate(ZW).split()
        start = rng.randint(0, max(0, len(words) - 1))
        text = recgen.join_to_budget(rng, words[start:], budget)
    else:
        s = SCRIPTS[rng.choice(list(SCRIPTS))]
        text = recgen.join_to_budget(rng, [_rand_word(rng, s) for _ in range(rng.randint(1, 5))], budget)
    return text, text  # Indic: render and label are both the logical string


def _build_spec() -> recgen.Spec:
    fonts = tuple(sorted(set(
        f for s in SCRIPTS.values() for f in recgen.discover_fonts(s["lang"])
    ) | set(recgen.discover_fonts("en"))))
    return recgen.Spec(name="indic", fonts=fonts, charset=CHARSET, reorder=True, gen_pair=gen_pair)


if __name__ == "__main__":
    recgen.run_cli(_build_spec())

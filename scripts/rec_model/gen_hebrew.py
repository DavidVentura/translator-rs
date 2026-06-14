"""Synthetic Hebrew text-line strips for PP-OCRv6 rec fine-tuning.

Hebrew is RTL with whole-line reordering, which a monotonic CTC can't learn from
logical labels — so the label is the VISUAL-order string (python-bidi get_display)
and we render that with no further reordering (recgen Spec.reorder=False). Inference
recovers logical order downstream (run-grouped bidi). Shared rendering/degradation/
CLI live in recgen.py.

  python gen_hebrew.py --out /tmp/heb --n 2000 --corpus hebrew_corpus.txt
  python gen_hebrew.py --out /tmp/heb-insp --n 24 --inspect --corpus hebrew_corpus.txt
"""

import random

from bidi import get_display

import recgen

HEBREW_LETTERS = "".join(chr(c) for c in range(0x05D0, 0x05EB))  # א..ת incl finals
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
PUNCT = " .,:;!?'\"()[]/%-+&@#₪€$"
HEBREW_PUNCT = "־׳״"  # maqaf, geresh, gershayim
CHARSET = HEBREW_LETTERS + LATIN + DIGITS + PUNCT + HEBREW_PUNCT

HEBREW_WORDS = (
    "של את על לא כן אני אתה הוא היא אנחנו הם זה זאת מה מי איפה מתי למה איך כמה יש "
    "אין היה יהיה עכשיו היום מחר אתמול שלום תודה בבקשה סליחה כסף בית עיר מדינה "
    "ישראל ירושלים תל אביב רחוב מספר טלפון דואר חשבון תאריך שעה דקה שנה חודש שבוע "
    "יום ראשון שני שלישי גדול קטן חדש ישן טוב רע יפה מים אוכל ספר מחשב עבודה "
    "משפחה ילד ילדה איש אישה מורה תלמיד בוקר ערב לילה"
).split()
LATIN_TOKENS = ("WiFi", "Email", "USB", "PDF", "OK", "Tel", "Fax", "App", "GPS", "TV")


def _gen_number(rng):
    k = rng.random()
    if k < 0.3:
        return f"{rng.randint(0, 31):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2026)}"
    if k < 0.5:
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if k < 0.7:
        return f"{rng.randint(1, 9999)}₪"
    if k < 0.85:
        return f"{rng.randint(1, 100)}%"
    return str(rng.randint(0, 999999))


def gen_logical(rng, corpus):
    budget = rng.randint(6, recgen.MAX_LABEL_LEN)
    if corpus and rng.random() < 0.7:
        words = rng.choice(corpus).split()
        start = rng.randint(0, max(0, len(words) - 1))
        return recgen.join_to_budget(rng, words[start:], budget)
    tokens = []
    for _ in range(8):
        r = rng.random()
        if r < 0.62:
            tokens.append(rng.choice(HEBREW_WORDS))
        elif r < 0.74:
            tokens.append(_gen_number(rng))
        elif r < 0.82:
            tokens.append(rng.choice(LATIN_TOKENS))
        else:
            tokens.append("".join(rng.choice(HEBREW_LETTERS) for _ in range(rng.randint(2, 6))))
    return recgen.join_to_budget(rng, tokens, budget)


def gen_pair(rng: random.Random, corpus: list[str]) -> tuple[str, str]:
    visual = get_display(gen_logical(rng, corpus), base_dir="R")
    return visual, visual  # render and label are both the visual-order string


SPEC = recgen.Spec(
    name="heb",
    fonts=recgen.discover_fonts("he"),
    charset=CHARSET,
    reorder=False,
    gen_pair=gen_pair,
)

if __name__ == "__main__":
    # Re-bidi the visual label back to logical only for legible inspect annotation.
    recgen.run_cli(SPEC, annotate=lambda s: get_display(s, base_dir="R"))

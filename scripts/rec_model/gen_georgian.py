"""Synthetic Georgian text-line strips for PP-OCRv6 rec fine-tuning.

Georgian is LTR with no bidi, no conjuncts and no pre-base matras, so the render text
and the label are the same logical string and HarfBuzz needs no reordering (recgen
Spec.reorder=False). This is gen_hebrew.py without the visual-order step.

The one Georgian wrinkle is case. Mkhedruli (U+10D0..) is caseless, but all-caps display
text is written in Mtavruli (U+1C90..) — a SEPARATE codepoint range with its own glyph
shapes (uniform height, no ascenders/descenders), not a styling of Mkhedruli. Mtavruli
therefore gets its own CTC classes and the label keeps whatever is on the page; see the
uppercase pass in `gen_pair` for how Mtavruli reaches the training set at all.

  python gen_georgian.py --out /tmp/geo --n 2000 --corpus georgian_corpus.bal.txt --dict georgian_latin_dict.txt
  python gen_georgian.py --out /tmp/geo-insp --n 24 --inspect --corpus georgian_corpus.bal.txt --dict georgian_latin_dict.txt
"""

import os
import random
import unicodedata as ud
from functools import lru_cache

import recgen

# ა..ჰ — the 33 letters of the modern alphabet. U+10F1..U+10FA continue the block with
# letters dropped from the alphabet in the 19th century; they stay out of BASE so the
# corpus frequency trim in build_corpus decides whether they earn a class.
MKHEDRULI_MODERN = "".join(chr(c) for c in range(0x10D0, 0x10F1))
MKHEDRULI_ARCHAIC = "".join(chr(c) for c in range(0x10F1, 0x10FB))
# The whole Mkhedruli block case-maps 1:1 onto Mtavruli (+0xBC0), archaic letters included.
MTAVRULI_MODERN = MKHEDRULI_MODERN.upper()
MTAVRULI_ARCHAIC = MKHEDRULI_ARCHAIC.upper()

LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
COMMON_PUNCT = " .,:;!?'\"()/-"
# Rare in cleaned wiki/news prose but real on signage, price lists and documents — kept
# regardless of corpus frequency and synth-covered to the training floor: brackets/symbols,
# currency (₾ lari is on every Georgian price tag), and the Georgian paragraph separator.
KEEP_PUNCT = "[]%+&@#€$₾"
GEORGIAN_PUNCT = "჻"

BASE = frozenset(MKHEDRULI_MODERN + MTAVRULI_MODERN + LATIN + DIGITS + COMMON_PUNCT)
KEEP_SET = frozenset(KEEP_PUNCT) | frozenset(GEORGIAN_PUNCT)

# All-caps rates. Wiki/news prose is ~100% Mkhedruli while shop fronts, banners and road
# signs — the live-camera case — are heavily Mtavruli, so without this pass the 33 Mtavruli
# classes would be trained only on synthetic filler. Uppercasing real corpus text instead
# puts them in genuine word contexts (IMPROVEMENTS.md: coverage from varied real sentences,
# never repetition). Line = a whole banner; word = one emphasised word inside running text.
MTAVRULI_LINE_FRAC = 0.12
MTAVRULI_WORD_FRAC = 0.08

GEORGIAN_WORDS = (
    "და არის ეს ის მე შენ ჩვენ თქვენ რომ მაგრამ თუ როგორ რა ვინ სად როდის რატომ "
    "რამდენი დიახ არა კი გამარჯობა მადლობა ნახვამდის ბოდიში უკაცრავად თბილისი "
    "საქართველო ქართული ენა ქალაქი ქუჩა სახლი ბინა წელი თვე დღე ღამე დილა საღამო "
    "შუადღე დრო საათი წუთი წამი ფული ლარი თეთრი მაღაზია რესტორანი კაფე სასტუმრო "
    "საავადმყოფო აფთიაქი სკოლა უნივერსიტეტი წიგნი მასწავლებელი სტუდენტი მოსწავლე "
    "კაცი ქალი ბავშვი ოჯახი მეგობარი ძმა დედა მამა წყალი პური ღვინო ჩაი ყავა "
    "საჭმელი ხორცი თევზი ხილი ბოსტნეული დიდი პატარა ახალი ძველი კარგი ცუდი ლამაზი "
    "მაღალი დაბალი ნომერი ტელეფონი მისამართი შესასვლელი გასასვლელი ღია დაკეტილი "
    "ავტობუსი მატარებელი მანქანა თვითმფრინავი გზა ხიდი ბანკი ფოსტა ეკლესია მუზეუმი "
    "ბაზარი თეატრი ბიბლიოთეკა ინფორმაცია ყურადღება აკრძალულია ფრთხილად გაყიდვა "
    "ფასდაკლება ჟურნალი გაზეთი ჰაერი ჰოსპიტალი ძალიან ცოტა ბევრი ხალხი სამსახური "
    "სამზარეულო ეზო ბაღი მთა ზღვა მდინარე ტყე ქვეყანა მსოფლიო ჭიშკარი ჭურჭელი "
    "ჟანრი ჰანგი ჩრდილოეთი სამხრეთი აღმოსავლეთი დასავლეთი"
).split()
LATIN_TOKENS = ("WiFi", "Email", "USB", "PDF", "OK", "Tel", "Fax", "App", "GPS", "TV")


def _assigned(ch: str) -> bool:
    try:
        ud.name(ch)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=1)
def _fonts() -> tuple[str, ...]:
    # Latin-only faces ride along so pure-Latin lines get the design variety the nine
    # Georgian families cannot supply; recgen.fonts_for still restricts any line carrying
    # a Georgian glyph to the faces that actually cover it.
    return tuple(sorted(set(recgen.discover_fonts("ka")) | set(recgen.discover_fonts("en"))))


@lru_cache(maxsize=1)
def candidate_charset() -> frozenset:
    """Every glyph the Georgian model could emit: base + curated keep-set + the archaic
    Mkhedruli/Mtavruli tails, restricted to assigned codepoints at least one discovered font
    renders. The archaic letters are offered to build_corpus's frequency trim rather than
    kept outright; unrenderable or unassigned codepoints are dropped here so they never
    become dead CTC classes acting as confusable sinks."""
    sup = "".join(sorted(BASE | KEEP_SET | set(MKHEDRULI_ARCHAIC) | set(MTAVRULI_ARCHAIC)))
    covered = set()
    for p in _fonts():
        covered |= recgen._covered(p, sup)
    return frozenset(ch for ch in sup if ord(ch) in covered and _assigned(ch))


def line_script(line: str) -> str:
    return "geor"  # single script: build_corpus treats the whole corpus as one bucket


CLOSER = {"[": "]", "(": ")", '"': '"'}
OPENER = {c: o for o, c in CLOSER.items()}
SENTENCE_END = ".!?"
CLAUSE_MID = ",;:"


def synth_tail(glyph: str, rng: random.Random, kept: frozenset) -> str:
    """A short line containing `glyph`, placed where the language actually puts it.

    build_corpus calls this only for glyphs the corpus leaves under the training floor, so a
    glyph the corpus never supplies at all (`@`, `#`, `[`, `]`, `჻` here) takes 100% of its
    exposure from this function. Position is then taught entirely by these lines, and a
    plausible-looking but wrong slot becomes a rule the model believes: Hebrew round 1 seeded
    final forms into synthetic mid-word contexts and corrupted the ס/ם boundary that way.
    Every branch reproduces the symbol's real syntactic slot; an unhandled glyph raises
    rather than falling into a generic "between two words" placement, because that placement
    is wrong for every symbol that is not a letter.
    """
    words = [w for w in GEORGIAN_WORDS if set(w) <= kept] or GEORGIAN_WORDS

    def geo() -> str:
        return rng.choice(words)

    def lat(lo: int = 2, hi: int = 6) -> str:
        return "".join(rng.choice(LATIN) for _ in range(rng.randint(lo, hi)))

    if glyph in MTAVRULI_MODERN or glyph in MTAVRULI_ARCHAIC:
        # Real words carrying the Mkhedruli counterpart, uppercased whole — the sign-shaped
        # context Mtavruli actually appears in. Random letter salad here would teach the
        # NRTR head a sequence prior that no Georgian text follows.
        low = glyph.lower()
        with_glyph = [w for w in words if low in w] or [low]
        return " ".join(rng.choice(with_glyph) for _ in range(rng.randint(1, 3))).upper()
    if glyph in MKHEDRULI_MODERN or glyph in MKHEDRULI_ARCHAIC:
        w = geo()  # a letter belongs inside a word, not welded between two of them
        cut = rng.randint(0, len(w))
        return f"{geo()} {w[:cut]}{glyph}{w[cut:]}"
    if ud.category(glyph) == "Sc" or glyph == "%":  # trails an amount: 25₾, 40%
        return f"{geo()} {rng.randint(1, 9999)}{glyph}"
    if glyph == "@":  # exists only inside an address
        return f"{lat(3, 8)}@{lat(3, 6)}.ge"
    if glyph == "#":  # leads the thing it numbers
        return f"{geo()} #{rng.randint(1, 999)}"
    if glyph in CLOSER or glyph in OPENER:
        # Brackets and quotes occur in pairs. Emitting one alone teaches an opener that never
        # closes, so the line carries both and whichever half was asked for is covered.
        o = glyph if glyph in CLOSER else OPENER[glyph]
        return f"{geo()} {o}{geo()}{CLOSER[o]} {geo()}"
    if glyph in SENTENCE_END:  # closes a clause, then a space
        return f"{geo()} {geo()}{glyph} {geo()} {geo()}"
    if glyph in CLAUSE_MID:  # binds to the preceding word, space after
        return f"{geo()}{glyph} {geo()} {geo()}"
    if glyph == "-":  # joins two words, or spans a numeric range
        return f"{geo()}-{geo()}" if rng.random() < 0.6 else f"{rng.randint(1, 99)}-{rng.randint(100, 999)}"
    if glyph == "/":  # separates alternatives or the parts of a date
        return f"{geo()}/{geo()}" if rng.random() < 0.5 else f"{rng.randint(1, 31)}/{rng.randint(1, 12)}"
    if glyph == "+":  # dial code, or between figures
        return f"+995{rng.randint(100000000, 999999999)}" if rng.random() < 0.6 else f"{rng.randint(1, 99)}+{rng.randint(1, 99)}"
    if glyph == "&":  # joins two Latin names, as on a shopfront
        return f"{lat()} & {lat()}"
    if glyph == "჻":  # paragraph separator sits between clauses
        return f"{geo()} {glyph} {geo()}"
    if glyph == "'":  # apostrophe inside transliterated Latin (Kartl'is, Sak'art'velo)
        return f"{lat(1, 3)}{glyph}{lat(2, 4)}"
    if glyph in LATIN or glyph in DIGITS:
        return f"{geo()} {glyph}{lat(1, 5)}"
    raise ValueError(f"synth_tail: no syntactic slot defined for {glyph!r} (U+{ord(glyph):04X})")


def _gen_number(rng: random.Random) -> str:
    k = rng.random()
    if k < 0.3:
        return f"{rng.randint(1, 31):02d}.{rng.randint(1, 12):02d}.{rng.randint(1990, 2026)}"
    if k < 0.5:
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if k < 0.7:
        return f"{rng.randint(1, 9999)}₾"
    if k < 0.85:
        return f"{rng.randint(1, 100)}%"
    return str(rng.randint(0, 999999))


def _gen_line(rng: random.Random, corpus: list[str]) -> str:
    budget = rng.randint(6, recgen.MAX_LABEL_LEN)
    if corpus and rng.random() < 0.7:
        words = rng.choice(corpus).split()
        start = rng.randint(0, max(0, len(words) - 1))
        return recgen.join_to_budget(rng, words[start:], budget)
    tokens = []
    for _ in range(8):
        r = rng.random()
        if r < 0.66:
            tokens.append(rng.choice(GEORGIAN_WORDS))
        elif r < 0.80:
            tokens.append(_gen_number(rng))
        else:
            tokens.append(rng.choice(LATIN_TOKENS))
    return recgen.join_to_budget(rng, tokens, budget)


def _upcase(text: str, rng: random.Random, vocab: frozenset) -> str:
    """All-caps `text`, or one word of it, when the result stays inside the model's classes.

    An archaic Mkhedruli letter can clear the corpus frequency trim while its Mtavruli
    counterpart — absent from prose, absent from BASE — does not, so uppercasing blindly
    would put a glyph in the label that the head has no output class for.
    """
    r = rng.random()
    if r < MTAVRULI_LINE_FRAC:
        candidate = text.upper()
    elif r < MTAVRULI_LINE_FRAC + MTAVRULI_WORD_FRAC:
        words = text.split(" ")
        i = rng.randrange(len(words))
        candidate = " ".join(w.upper() if j == i else w for j, w in enumerate(words))
    else:
        return text
    return candidate if set(candidate) - {" "} <= vocab else text


def gen_pair(rng: random.Random, corpus: list[str], vocab: frozenset) -> tuple[str, str]:
    text = _upcase(_gen_line(rng, corpus), rng, vocab)
    return text, text  # Georgian is LTR and unshaped: render and label are one string


_MKHEDRULI = MKHEDRULI_MODERN + MKHEDRULI_ARCHAIC
_CAPS_REMAP = tuple(
    [(m, k) for k, m in zip(_MKHEDRULI, MTAVRULI_MODERN + MTAVRULI_ARCHAIC)]
    + [(k, None) for k in _MKHEDRULI]
)


def _caps_remap(fonts: tuple[str, ...]) -> dict[str, tuple[tuple[str, str | None], ...]]:
    """Faces that draw caps-shaped Georgian at the Mkhedruli codepoints, mapped onto Mtavruli.

    BPG's Caps families (Excelsior Caps, Nateli Caps, Mrgvlovani Caps) date from 2012, before
    Unicode 11 gave Mtavruli its own block, so all-caps Georgian was set by swapping to a caps
    font while the text stayed in Mkhedruli codepoints. Measured against Noto across the
    alphabet, their glyphs resemble Mtavruli more than Mkhedruli, so they are routed to
    Mtavruli labels and barred from Mkhedruli — otherwise the same caps shapes would train
    under both labels and blunt the distinction the separate classes exist to draw.
    """
    return {p: _CAPS_REMAP for p in fonts
            if "caps" in os.path.basename(p).lower()
            and recgen._ft_face(p).get_char_index(0x10D0)
            and not recgen._ft_face(p).get_char_index(0x1C90)}


def _build_spec() -> recgen.Spec:
    fonts = _fonts()
    return recgen.Spec(
        name="geor", fonts=fonts, charset="".join(sorted(candidate_charset())),
        reorder=False, gen_pair=gen_pair, font_remap=_caps_remap(fonts),
    )


if __name__ == "__main__":
    recgen.run_cli(_build_spec())

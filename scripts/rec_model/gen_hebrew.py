"""Synthetic Hebrew text-line strips for PP-OCRv6 recognizer fine-tuning.

The recognizer is a visual left-to-right CTC transcriber with no notion of
direction (see train_paddle_rec.md). So every line is produced as:

  logical text -> python-bidi get_display -> VISUAL-ORDER string

The visual string is BOTH what we render and the training label, which keeps the
label monotonic with the left-to-right image scan even for mixed Hebrew+Latin.
Recovering logical order is an inference-time post-process in ppocr.rs, not this
generator's job.

Shaping is explicit HarfBuzz (uharfbuzz) + FreeType rasterization, not naive PIL
placement. The visual string is already reordered, so it shapes LTR; Hebrew has
no cursive joining, so per-character glyphs render correctly. Any .notdef glyph
(gid 0) rejects the sample, so tofu never reaches training. Arabic/Indic will
need per-run directional shaping on top of this — deferred to their milestones.

Output is PaddleOCR rec format: images under <out>/images and a <out>/labels.txt
of `images/NAME.png\tLABEL` lines, plus a keys.txt of the charset seen.

CLI:
  python gen_hebrew.py --out /tmp/heb-rec --n 2000
  python gen_hebrew.py --out /tmp/heb-inspect --n 24 --inspect   # annotated sheets
  python gen_hebrew.py --out ... --corpus hebrew_lines.txt        # realistic text
"""

import argparse
import io
import os
import random
import subprocess
from functools import lru_cache

import freetype
import numpy as np
import uharfbuzz as hb
from bidi import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

STRIP_HEIGHT = 48
# PP-OCRv6 rec config caps lines at max_text_length: 25 (use_space_char counts
# spaces). Longer real lines are split by detection, so training lines stay short.
MAX_LABEL_LEN = 25

# Hebrew letters incl. final forms; ASCII letters/digits/punct for mixed content;
# shekel + common symbols. The real fine-tune dict is the v6 small charset plus
# the Hebrew block; this is what the generator can actually produce.
HEBREW_LETTERS = "".join(chr(c) for c in range(0x05D0, 0x05EB))  # א..ת incl finals
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
PUNCT = " .,:;!?'\"()[]/%-+&@#₪€$"
HEBREW_PUNCT = "־׳״"  # maqaf, geresh, gershayim — real Hebrew orthography
CHARSET = HEBREW_LETTERS + LATIN + DIGITS + PUNCT + HEBREW_PUNCT

# Common Hebrew words for realistic-ish lines when no corpus is given. Real text
# matters for the NRTR language-model head; pass --corpus for production data.
HEBREW_WORDS = (
    "של את על לא כן אני אתה הוא היא אנחנו הם זה זאת מה מי איפה מתי למה איך כמה יש "
    "אין היה יהיה עכשיו היום מחר אתמול שלום תודה בבקשה סליחה כסף בית עיר מדינה "
    "ישראל ירושלים תל אביב רחוב מספר טלפון דואר חשבון תאריך שעה דקה שנה חודש שבוע "
    "יום ראשון שני שלישי גדול קטן חדש ישן טוב רע יפה מים אוכל ספר מחשב עבודה "
    "משפחה ילד ילדה איש אישה מורה תלמיד בוקר ערב לילה"
).split()

LATIN_TOKENS = ("WiFi", "Email", "USB", "PDF", "OK", "Tel", "Fax", "App", "GPS", "TV")


@lru_cache(maxsize=1)
def hebrew_fonts() -> tuple[str, ...]:
    out = subprocess.run(
        ["fc-list", ":lang=he", "file"], capture_output=True, text=True, check=True
    ).stdout
    paths = []
    for line in out.splitlines():
        p = line.split(":")[0].strip()
        if p.lower().endswith((".ttf", ".otf")):
            paths.append(p)
    if not paths:
        raise RuntimeError("fc-list found no Hebrew fonts")
    return tuple(sorted(set(paths)))


@lru_cache(maxsize=256)
def _ft_face(path: str) -> freetype.Face:
    return freetype.Face(path)


@lru_cache(maxsize=256)
def _hb_font(path: str) -> hb.Font:
    return hb.Font(hb.Face(hb.Blob.from_file_path(path)))


@lru_cache(maxsize=256)
def _covered(path: str) -> frozenset:
    face = _ft_face(path)
    return frozenset(cp for cp in map(ord, CHARSET) if face.get_char_index(cp) != 0)


def fonts_for(text: str) -> list[str]:
    need = set(map(ord, text)) - {0x20}
    return [p for p in hebrew_fonts() if need <= _covered(p)]


def _join_to_budget(rng: random.Random, words: list[str], budget: int) -> str:
    """Greedily join words (space-separated) without exceeding `budget` chars.

    bidi reordering preserves character count, so capping the logical line caps
    the visual label to PP-OCRv6's max_text_length.
    """
    out: list[str] = []
    length = 0
    for w in words:
        add = len(w) + (1 if out else 0)
        if length + add > budget:
            break
        out.append(w)
        length += add
    return " ".join(out) if out else words[0][:budget]


def gen_logical(rng: random.Random, corpus: list[str]) -> str:
    """One logical (reading-order) line within MAX_LABEL_LEN: corpus span or word mix."""
    budget = rng.randint(6, MAX_LABEL_LEN)
    if corpus and rng.random() < 0.7:
        words = rng.choice(corpus).split()
        start = rng.randint(0, max(0, len(words) - 1))
        return _join_to_budget(rng, words[start:], budget)
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
    return _join_to_budget(rng, tokens, budget)


def _gen_number(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.3:
        return f"{rng.randint(0, 31):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2026)}"
    if kind < 0.5:
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if kind < 0.7:
        return f"{rng.randint(1, 9999)}₪"
    if kind < 0.85:
        return f"{rng.randint(1, 100)}%"
    return str(rng.randint(0, 999999))


def shape_render(visual: str, font_path: str, px: int) -> np.ndarray | None:
    """Coverage (H x W float 0..1) for the already-visual-order string, or None on tofu.

    Shapes LTR with HarfBuzz (the string is pre-reordered) and rasterizes each
    glyph with FreeType at the shaped pen positions. Returns None if any glyph is
    .notdef so the caller can retry with another font/text.
    """
    hbfont = _hb_font(font_path)
    hbfont.scale = (px * 64, px * 64)
    buf = hb.Buffer()
    buf.add_str(visual)
    buf.guess_segment_properties()
    buf.direction = "ltr"
    hb.shape(hbfont, buf)
    infos, poss = buf.glyph_infos, buf.glyph_positions
    if any(i.codepoint == 0 for i in infos):
        return None

    ft = _ft_face(font_path)
    ft.set_pixel_sizes(0, px)
    total_adv = sum(p.x_advance for p in poss) // 64
    pad = max(4, px // 6)
    width = total_adv + 2 * pad
    height = int(px * 1.7)
    baseline = int(px * 1.3)
    canvas = np.zeros((height, width), np.float32)
    x = pad * 64
    for info, pos in zip(infos, poss):
        ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bm = ft.glyph.bitmap
        gw, gh = bm.width, bm.rows
        if gw and gh:
            glyph = np.asarray(bm.buffer, np.uint8).reshape(gh, gw).astype(np.float32) / 255.0
            gx = (x + pos.x_offset) // 64 + ft.glyph.bitmap_left
            gy = baseline - ft.glyph.bitmap_top - pos.y_offset // 64
            y0, x0 = max(0, gy), max(0, gx)
            y1, x1 = min(height, gy + gh), min(width, gx + gw)
            if y1 > y0 and x1 > x0:
                sub = glyph[y0 - gy : y1 - gy, x0 - gx : x1 - gx]
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)
        x += pos.x_advance
    rows = np.where(canvas.max(axis=1) > 0.05)[0]
    cols = np.where(canvas.max(axis=0) > 0.05)[0]
    if rows.size < 4 or cols.size < 4:
        return None
    m = pad
    r0, r1 = max(0, rows[0] - m), min(height, rows[-1] + 1 + m)
    c0, c1 = max(0, cols[0] - m), min(width, cols[-1] + 1 + m)
    return canvas[r0:r1, c0:c1]


def random_color(rng: random.Random) -> np.ndarray:
    return np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)


def background(rng: random.Random, h: int, w: int) -> np.ndarray:
    """H x W x 3 background in 0..1: solid, gradient, or low-frequency texture."""
    kind = rng.random()
    c0 = random_color(rng)
    if kind < 0.45:
        return np.broadcast_to(c0, (h, w, 3)).copy()
    if kind < 0.8:
        c1 = random_color(rng)
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.cos(angle) * xx / max(w, 1) + np.sin(angle) * yy / max(h, 1)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        return t[..., None] * c1 + (1 - t[..., None]) * c0
    grid = rng.choice([6, 10, 16])
    blob = np.random.default_rng(rng.getrandbits(32)).random(
        (max(2, h // grid), max(2, w // grid), 3)
    ).astype(np.float32)
    blob = np.asarray(
        Image.fromarray((blob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        np.float32,
    ) / 255.0
    mix = rng.uniform(0.4, 0.85)
    return mix * blob + (1 - mix) * c0


def degrade(img: np.ndarray, rng: random.Random, native_h: int) -> np.ndarray:
    """Camera/scan degradations on the composited line; blur scales with text height."""
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if rng.random() < 0.5:
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, min(0.4 + native_h * 0.03, 1.8))))
    if rng.random() < 0.3:
        s = rng.uniform(0.55, 0.85)
        small = pil.resize((max(8, int(pil.width * s)), max(8, int(pil.height * s))), Image.BILINEAR)
        pil = small.resize((pil.width, pil.height), Image.BILINEAR)
    if rng.random() < 0.75:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=rng.randint(40, 95))
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    out = np.asarray(pil, np.float32) / 255.0
    if rng.random() < 0.5:
        yy, xx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]].astype(np.float32)
        angle = rng.uniform(0, 2 * np.pi)
        t = np.cos(angle) * xx / out.shape[1] + np.sin(angle) * yy / out.shape[0]
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        out = out * np.clip(rng.uniform(0.6, 1.0) + t * rng.uniform(0.0, 0.4), 0.4, 1.2)[..., None]
    if rng.random() < 0.6:
        out = out + np.random.default_rng(rng.getrandbits(32)).normal(
            0, rng.uniform(0.005, 0.035), out.shape
        ).astype(np.float32)
    return np.clip(out, 0, 1)


def compose(rng: random.Random, cov: np.ndarray, native_h: int) -> np.ndarray:
    """Color the coverage over a background with enforced contrast, then degrade."""
    h, w = cov.shape
    bg = background(rng, h, w)
    ink = random_color(rng)
    mean_bg = float(bg.mean())
    # Push ink away from the background mean so the line stays legible after degrade.
    if mean_bg > 0.5:
        ink = ink * rng.uniform(0.0, 0.35)
    else:
        ink = 1.0 - (1.0 - ink) * rng.uniform(0.0, 0.35)
    img = cov[..., None] * ink + (1 - cov[..., None]) * bg
    return degrade(img, rng, native_h)


def sample(rng: random.Random, corpus: list[str]) -> tuple[np.ndarray, str] | None:
    """One (H=48 RGB uint8, visual-order label) pair, or None after retries."""
    for _ in range(8):
        logical = gen_logical(rng, corpus)
        visual = get_display(logical, base_dir="R")
        if not visual.strip():
            continue
        fonts = fonts_for(visual)
        if not fonts:
            continue
        native_h = rng.randint(22, 60)
        px = max(8, int(native_h * rng.uniform(0.62, 0.9)))
        cov = shape_render(visual, rng.choice(fonts), px)
        if cov is None or cov.shape[1] < 8:
            continue
        # Fit the rendered text into a native strip of height native_h.
        strip_h = max(native_h, cov.shape[0])
        top = (strip_h - cov.shape[0]) // 2
        strip = np.zeros((strip_h, cov.shape[1] + rng.randint(2, max(3, native_h))), np.float32)
        lpad = rng.randint(1, max(2, strip.shape[1] - cov.shape[1] - 1))
        strip[top : top + cov.shape[0], lpad : lpad + cov.shape[1]] = cov
        img = compose(rng, strip, strip_h)
        out_w = max(16, round(img.shape[1] * STRIP_HEIGHT / img.shape[0]))
        if out_w > 1200:
            continue
        img = np.asarray(
            Image.fromarray((img * 255).astype(np.uint8)).resize((out_w, STRIP_HEIGHT), Image.BILINEAR),
            np.uint8,
        )
        return img, visual
    return None


def write_inspect_sheet(samples: list[tuple[np.ndarray, str]], path: str) -> None:
    band = 18
    rows = [np.asarray(im, np.uint8) for im, _ in samples]
    width = max(r.shape[1] for r in rows)
    annot = ImageFont.load_default()
    cells = []
    for (im, label), r in zip(samples, rows):
        cell = np.full((STRIP_HEIGHT + band, width, 3), 255, np.uint8)
        cell[band:, : r.shape[1]] = r
        pim = Image.fromarray(cell)
        # Re-bidi the visual label back to logical just for legible annotation.
        ImageDraw.Draw(pim).text((2, 2), get_display(label, base_dir="R"), fill=(200, 0, 0), font=annot)
        cells.append(np.asarray(pim))
    Image.fromarray(np.concatenate(cells, axis=0)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", help="UTF-8 file, one Hebrew line per row")
    ap.add_argument("--inspect", action="store_true", help="write an annotated QA sheet instead of a dataset")
    ap.add_argument("--prefix", default="", help="filename/label-file shard tag for parallel workers sharing one --out")
    args = ap.parse_args()

    corpus = []
    if args.corpus:
        with open(args.corpus, encoding="utf-8") as f:
            corpus = [ln.strip() for ln in f if ln.strip()]

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.inspect:
        got = []
        while len(got) < args.n:
            s = sample(rng, corpus)
            if s:
                got.append(s)
        write_inspect_sheet(got, os.path.join(args.out, "inspect.png"))
        print(f"wrote {os.path.join(args.out, 'inspect.png')} ({len(got)} samples)")
        return

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    tag = f"{args.prefix}_" if args.prefix else ""
    labels_path = os.path.join(args.out, f"labels{('_' + args.prefix) if args.prefix else ''}.txt")
    n = 0
    with open(labels_path, "w", encoding="utf-8") as labels:
        while n < args.n:
            s = sample(rng, corpus)
            if not s:
                continue
            img, label = s
            name = f"images/{tag}heb_{n:07d}.png"
            Image.fromarray(img).save(os.path.join(args.out, name))
            labels.write(f"{name}\t{label}\n")
            n += 1
            if n % 2000 == 0:
                print(f"  [{args.prefix or '0'}] {n}/{args.n}", flush=True)
    print(f"wrote {n} samples -> {labels_path}")


if __name__ == "__main__":
    main()

"""Synthetic training strips for the ink model.

Renders text with known per-pixel coverage (the label), composites it over
procedural backgrounds (solids, gradients, noise blobs — the colors.jpg cases),
then degrades the image only (JPEG, blur, noise, illumination) so the label
stays clean. Language is irrelevant for ink masks; glyph shapes are what
matters, so text mixes dictionary words with random charset strings for rare
glyph coverage.

CLI: python gen_data.py --out /tmp/ink-samples --n 16   (writes inspection PNGs)
Importable: sample() -> (image float32 HxWx3 in 0..1, coverage float32 HxW)
"""

import argparse
import io
import os
import random
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HEIGHT = 48
WIDTHS = list(range(96, 513, 16))

LATIN_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".,;:!?¡¿'\"()[]%&@#€$£+-*/=<>_"
    "àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿßœ"
    "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝŒ"
    "ąćęłńśźżĄĆĘŁŃŚŹŻčďěňřšťůžČĎĚŇŘŠŤŮŽőűŐŰ"
)

FONT_BLOCKLIST = (
    "emoji",
    "symbol",
    "dingbat",
    "music",
    "math",
    "braille",
    "awesome",
)


@lru_cache(maxsize=1)
def font_paths() -> list[str]:
    out = subprocess.run(
        ["fc-list", ":lang=en", "file"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = []
    for line in out.splitlines():
        p = line.split(":")[0].strip()
        low = p.lower()
        if not (low.endswith(".ttf") or low.endswith(".otf")):
            continue
        if any(b in low for b in FONT_BLOCKLIST):
            continue
        paths.append(p)
    if not paths:
        raise RuntimeError("fc-list found no usable fonts")
    return sorted(set(paths))


@lru_cache(maxsize=1)
def dictionary_words() -> list[str]:
    for path in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                words = [w.strip() for w in f if 2 <= len(w.strip()) <= 14]
            if words:
                return words
    return []


def random_text(rng: random.Random) -> str:
    words = []
    dictionary = dictionary_words()
    for _ in range(rng.randint(1, 8)):
        if dictionary and rng.random() < 0.6:
            w = rng.choice(dictionary)
            if rng.random() < 0.2:
                w = w.capitalize()
            elif rng.random() < 0.07:
                w = w.upper()
        else:
            n = rng.randint(1, 10)
            w = "".join(rng.choice(LATIN_CHARSET) for _ in range(n))
        words.append(w)
    return " ".join(words)


def render_coverage(
    rng: random.Random, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Antialiased glyph coverage on a height x width canvas.

    Returns `(total, fill)`: `total` is the union coverage including any outline
    stroke (this is the training label — outlines are ink); `fill` is the glyph
    core only, so the caller can give the outline ring its own color. They're
    identical when the text isn't outlined.
    """
    # Render oversized then rotate, so the rotation doesn't clip glyphs.
    pad = max(8, height // 2)
    canvas = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
    fill_canvas = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(canvas)
    fill_draw = ImageDraw.Draw(fill_canvas)
    jitter = max(1, height // 12)
    stroke = rng.randint(1, max(1, height // 10)) if rng.random() < 0.2 else 0
    for attempt in range(8):
        path = rng.choice(font_paths())
        size = rng.randint(max(7, int(height * 0.45)), max(8, int(height * 0.95)))
        try:
            font = ImageFont.truetype(path, size=size)
            text = random_text(rng)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            if right - left < 8 or bottom - top < 4:
                continue
        except OSError:
            continue
        x = pad + rng.randint(-jitter, 2 * jitter) - left
        y = pad + (height - (bottom - top)) // 2 + rng.randint(-jitter, jitter) - top
        draw.text((x, y), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
        fill_draw.text((x, y), text, font=font, fill=255)
        break
    if rng.random() < 0.5:
        angle = rng.uniform(-3.0, 3.0)
        center = (pad, pad + height // 2)
        canvas = canvas.rotate(angle, resample=Image.BILINEAR, center=center)
        fill_canvas = fill_canvas.rotate(angle, resample=Image.BILINEAR, center=center)
    total = np.asarray(canvas, dtype=np.float32) / 255.0
    fill = np.asarray(fill_canvas, dtype=np.float32) / 255.0
    return (
        total[pad : pad + height, pad : pad + width],
        fill[pad : pad + height, pad : pad + width],
    )


def random_color(rng: random.Random) -> np.ndarray:
    return np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)


def gradient_field(rng: random.Random, h: int, w: int) -> np.ndarray:
    """HxWx3 background in 0..1: solid, linear/radial gradient, or noise blobs."""
    kind = rng.random()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c0, c1 = random_color(rng), random_color(rng)
    if kind < 0.25:
        return np.broadcast_to(c0, (h, w, 3)).copy()
    if kind < 0.65:
        angle = rng.uniform(0, 2 * np.pi)
        t = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        if rng.random() < 0.3:
            t = np.abs(2 * t - 1)  # two-tone band through the strip
        return t[..., None] * c1 + (1 - t[..., None]) * c0
    if kind < 0.85:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = np.clip(r / max(r.max(), 1e-6), 0, 1)
        return t[..., None] * c1 + (1 - t[..., None]) * c0
    blob = np.random.default_rng(rng.getrandbits(32)).random((h // 8, w // 8, 3)).astype(np.float32)
    blob = np.asarray(
        Image.fromarray((blob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return 0.5 * blob + 0.5 * c0


def degrade(img: np.ndarray, rng: random.Random, native_h: int) -> np.ndarray:
    """Camera/screen degradations applied to the composited image only.

    Blur scales with the native text height: a 1.8 px gaussian erases 12 px text
    outright but is realistic camera softness on 40 px text. Det/rec gate what
    reaches the ink model, so training must not contain text they would reject.
    """
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if rng.random() < 0.5:
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.35 + native_h * 0.032)))
    if rng.random() < 0.25:
        # Crude motion blur: directional box kernel.
        k = 3 if native_h < 24 else rng.choice([3, 5])
        kernel = [0.0] * (k * k)
        if rng.random() < 0.5:
            for i in range(k):
                kernel[(k // 2) * k + i] = 1.0 / k
        else:
            for i in range(k):
                kernel[i * k + k // 2] = 1.0 / k
        pil = pil.filter(ImageFilter.Kernel((k, k), kernel, scale=1.0))
    if rng.random() < 0.35:
        scale = rng.uniform(0.4, 0.8)
        small = pil.resize(
            (max(8, int(pil.width * scale)), max(8, int(pil.height * scale))), Image.BILINEAR
        )
        pil = small.resize((pil.width, pil.height), Image.BILINEAR)
    if rng.random() < 0.7:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=rng.randint(30, 95))
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    out = np.asarray(pil, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        angle = rng.uniform(0, 2 * np.pi)
        t = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        shade = rng.uniform(0.55, 1.0) + t * rng.uniform(0.0, 0.45)
        out = out * np.clip(shade, 0.4, 1.2)[..., None]
    if rng.random() < 0.6:
        out = out + np.random.default_rng(rng.getrandbits(32)).normal(
            0, rng.uniform(0.005, 0.04), out.shape
        ).astype(np.float32)
    if rng.random() < 0.3:
        lo, hi = rng.uniform(0.0, 0.15), rng.uniform(0.75, 1.0)
        out = out * (hi - lo) + lo
    return np.clip(out, 0, 1)


def legible(img: np.ndarray, cov: np.ndarray) -> bool:
    """Reject pairs whose degraded ink no longer contrasts with its background.

    Det/rec sit upstream of the ink model at inference, so unreadable text never
    reaches it — training on it would teach the model to hallucinate ink. Compare
    median ink color against median background color within the text's own rows
    (global background medians lie on gradient strips).
    """
    ink_mask = cov > 0.6
    if ink_mask.sum() < 30:
        return False
    text_rows = ink_mask.any(axis=1)
    bg_mask = (cov < 0.05) & text_rows[:, None]
    if bg_mask.sum() < 30:
        return False
    d = np.abs(np.median(img[ink_mask], axis=0) - np.median(img[bg_mask], axis=0))
    return float(d.max()) > 0.12


def sample(rng: random.Random | None = None, width: int | None = None):
    """One training pair, generated at *native* scale then resized to height 48.

    Real strips are dewarped crops whose source text is mostly 12–30 px tall; the
    squash-to-48 widens and softens the antialiased glyph edges. Rendering directly
    at 48 px would teach the model edge statistics it never sees at inference, so we
    render/composite/degrade at a sampled native height and resample both image and
    label exactly like the pipeline does.
    """
    rng = rng or random.Random()
    width = width or rng.choice(WIDTHS)
    for _attempt in range(8):
        img, cov, ok = _sample_once(rng, width)
        if ok:
            break
    return img, cov


def _sample_once(rng: random.Random, width: int):
    # Tail above 48: signage/display text whose native height far exceeds the strip —
    # after the squash its strokes are very thick and glyph interiors are majority-ink,
    # a density regime body text never reaches (validation: station-sign letters
    # mottled when 12–48 was the whole training range).
    if rng.random() < 0.2:
        native_h = int(rng.uniform(48, 160))
    else:
        native_h = int(rng.triangular(12, 48, 20))
    native_w = max(16, round(width * native_h / HEIGHT))
    cov, fill = render_coverage(rng, native_w, native_h)
    bg = gradient_field(rng, native_h, native_w)
    if rng.random() < 0.15:
        # Drop shadow: an offset dark replica behind the glyphs. It is *not* ink —
        # the label stays `cov` — so the model learns to leave shadows to the
        # background reconstruction instead of matting them.
        dy, dx = rng.randint(1, 3), rng.randint(-2, 3)
        shadow = np.roll(np.roll(cov, dy, axis=0), dx, axis=1)
        bg = bg * (1.0 - (shadow * rng.uniform(0.35, 0.7))[..., None])
    ink = random_color(rng)
    # Sometimes force low contrast against the background mean.
    if rng.random() < 0.25:
        mean_bg = bg.mean(axis=(0, 1))
        ink = np.clip(mean_bg + np.sign(ink - mean_bg) * rng.uniform(0.12, 0.35), 0, 1)
    if rng.random() < 0.1:
        # Gradient ink: lerp between two ink colors across the strip.
        ink2 = random_color(rng)
        t = np.linspace(0, 1, native_w, dtype=np.float32)[None, :, None]
        ink_field = t * ink2 + (1 - t) * ink
    else:
        ink_field = np.broadcast_to(ink, (native_h, native_w, 3)).copy()
    ring = np.clip(cov - fill, 0.0, 1.0)
    if ring.max() > 0.05:
        # Outlined text: the ring gets its own color (e.g. white core, blue edge).
        outline = random_color(rng)
        frac = np.divide(ring, np.maximum(cov, 1e-3))[..., None]
        ink_field = ink_field * (1 - frac) + outline * frac
    img = cov[..., None] * ink_field + (1 - cov[..., None]) * bg
    img = degrade(img, rng, native_h)
    ok = legible(img, cov)
    if native_h != HEIGHT:
        img = np.asarray(
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
                (width, HEIGHT), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        cov = np.asarray(
            Image.fromarray(cov, mode="F").resize((width, HEIGHT), Image.BILINEAR),
            dtype=np.float32,
        )
    return (
        img.astype(np.float32, copy=False),
        np.clip(cov, 0.0, 1.0).astype(np.float32, copy=False),
        ok,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    for i in range(args.n):
        img, cov = sample(rng)
        h, w = cov.shape
        sheet = np.ones((h * 2 + 4, w, 3), dtype=np.float32)
        sheet[:h] = img
        sheet[h + 4 :] = cov[..., None]
        Image.fromarray((sheet * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"sample-{i:03d}.png")
        )
    print(f"wrote {args.n} samples to {args.out} (top: image, bottom: coverage label)")


if __name__ == "__main__":
    main()

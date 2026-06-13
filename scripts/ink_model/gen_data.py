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

# Unicode blocks for dense-script samples. Noto CJK (the :lang=ko fonts) covers all
# three fully, so generating glyphs straight from the ranges never hits .notdef.
CJK_BLOCKS = ((0xAC00, 0xD7A3), (0x4E00, 0x9FFF), (0x3040, 0x30FF))

# Filename markers for heavy weights. The random font pool is mostly regular weight,
# so display-style superbold strokes are under-trained and the matte under-covers them.
HEAVY_TOKENS = ("bold", "black", "heavy", "extrabold", "extrablack", "semibold", "-bd", "-blk")


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
def cjk_font_paths() -> list[str]:
    out = subprocess.run(
        ["fc-list", ":lang=ko", "file"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = []
    for line in out.splitlines():
        p = line.split(":")[0].strip()
        if p.lower().endswith((".ttf", ".otf", ".ttc")):
            paths.append(p)
    if not paths:
        raise RuntimeError("fc-list found no CJK fonts")
    return sorted(set(paths))


@lru_cache(maxsize=2)
def _heavy_fonts(cjk: bool) -> tuple[str, ...]:
    base = cjk_font_paths() if cjk else font_paths()
    heavy = tuple(p for p in base if any(t in p.lower() for t in HEAVY_TOKENS))
    return heavy or tuple(base)


def cjk_text(rng: random.Random) -> str:
    lo, hi = rng.choice(CJK_BLOCKS)
    chars = []
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.15:  # Latin digits/units mix in, like "100% 30ml"
            chars.append(rng.choice("0123456789%."))
        else:
            chars.append(chr(rng.randint(lo, hi)))
    return "".join(chars)


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
    rng: random.Random, width: int, height: int, log: dict | None = None
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
    draw = ImageDraw.Draw(canvas)
    jitter = max(1, height // 12)
    stroke = rng.randint(1, max(1, height // 10)) if rng.random() < 0.2 else 0
    # The fill canvas (glyph core, no stroke) only differs from `total` when the text
    # is outlined; rendering glyphs is the per-sample hotspot, so skip the second draw
    # entirely when there's no stroke and alias `fill` to `total`.
    outlined = stroke > 0
    fill_canvas = Image.new("L", canvas.size, 0) if outlined else None
    fill_draw = ImageDraw.Draw(fill_canvas) if outlined else None
    # Dense-script (CJK) share: thick strokes with tiny enclosed counters are the one
    # regime Latin-only training floods (it marks the counters as ink). Real CJK
    # glyphs teach the model to keep them open.
    use_cjk = rng.random() < 0.25
    heavy = rng.random() < 0.4  # bias toward heavy/black weights for superbold strokes
    fonts = _heavy_fonts(use_cjk) if heavy else (cjk_font_paths() if use_cjk else font_paths())
    for attempt in range(8):
        path = rng.choice(fonts)
        size = rng.randint(max(7, int(height * 0.45)), max(8, int(height * 0.95)))
        try:
            font = ImageFont.truetype(path, size=size)
            text = cjk_text(rng) if use_cjk else random_text(rng)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            if right - left < 8 or bottom - top < 4:
                continue
        except OSError:
            continue
        x = pad + rng.randint(-jitter, 2 * jitter) - left
        y = pad + (height - (bottom - top)) // 2 + rng.randint(-jitter, jitter) - top
        draw.text((x, y), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
        if outlined:
            fill_draw.text((x, y), text, font=font, fill=255)
        break
    if rng.random() < 0.5:
        angle = rng.uniform(-3.0, 3.0)
        center = (pad, pad + height // 2)
        canvas = canvas.rotate(angle, resample=Image.BILINEAR, center=center)
        if outlined:
            fill_canvas = fill_canvas.rotate(angle, resample=Image.BILINEAR, center=center)
    total = np.asarray(canvas, dtype=np.float32) / 255.0
    fill = np.asarray(fill_canvas, dtype=np.float32) / 255.0 if outlined else total
    thick_k = 0
    if height >= 48 and rng.random() < 0.4:
        # Thicken to simulate super-heavy weight beyond what fonts provide: trains the
        # solid thick interiors the matte otherwise only outlines (edge-starvation).
        # Only on the large/display regime, with the kernel scaled to the text height —
        # any dilation on small body text merges 'i' dots, closes counters, and fuses
        # letters, corrupting the label and starving thin-stroke coverage.
        thick_k = int(np.clip(round(height / 40) * 2 + 1, 3, 11))
        total = np.asarray(
            Image.fromarray((total * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(thick_k)),
            dtype=np.float32) / 255.0
        if outlined:
            fill = np.asarray(
                Image.fromarray((fill * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(thick_k)),
                dtype=np.float32) / 255.0
        else:
            fill = total
    if log is not None:
        log.update(font=os.path.basename(path), size=size, cjk=use_cjk,
                   heavy=heavy, stroke=stroke, thick_k=thick_k)
    return (
        total[pad : pad + height, pad : pad + width],
        fill[pad : pad + height, pad : pad + width],
    )


@lru_cache(maxsize=64)
def _coord_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only (yy, xx) pixel grids, cached per size — callers only read them."""
    yy, xx = np.mgrid[0:h, 0:w]
    yy = np.ascontiguousarray(yy, dtype=np.float32)
    xx = np.ascontiguousarray(xx, dtype=np.float32)
    yy.flags.writeable = False
    xx.flags.writeable = False
    return yy, xx


def random_color(rng: random.Random) -> np.ndarray:
    return np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)


def gradient_field(rng: random.Random, h: int, w: int, log: dict | None = None) -> np.ndarray:
    """HxWx3 background in 0..1: solid, gradient, color blocks, or busy texture.

    The block and busy-texture cases are deliberately high-contrast with hard edges:
    saturated colored regions and torn-paper/shaded boundaries are exactly the
    backgrounds the model wrongly mattes as ink, so it must see them with an empty
    label and learn that structure alone isn't ink.
    """
    kind = rng.random()
    if log is not None:
        log["bg"] = ("solid" if kind < 0.20 else "linear" if kind < 0.50
                     else "radial" if kind < 0.62 else "blocks" if kind < 0.80 else "texture")
    yy, xx = _coord_grid(h, w)
    c0, c1 = random_color(rng), random_color(rng)
    if kind < 0.20:
        return np.broadcast_to(c0, (h, w, 3)).copy()
    if kind < 0.50:
        angle = rng.uniform(0, 2 * np.pi)
        t = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        if rng.random() < 0.3:
            t = np.abs(2 * t - 1)  # two-tone band through the strip
        return t[..., None] * c1 + (1 - t[..., None]) * c0
    if kind < 0.62:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = np.clip(r / max(r.max(), 1e-6), 0, 1)
        return t[..., None] * c1 + (1 - t[..., None]) * c0
    if kind < 0.80:
        field = np.broadcast_to(c0, (h, w, 3)).copy()
        for _ in range(rng.randint(2, 6)):
            bx0, bx1 = sorted(rng.randint(0, w) for _ in range(2))
            by0, by1 = sorted(rng.randint(0, h) for _ in range(2))
            field[by0:by1, bx0:bx1] = random_color(rng)
        return field
    grid = rng.choice([6, 8, 12])
    blob = np.random.default_rng(rng.getrandbits(32)).random(
        (max(2, h // grid), max(2, w // grid), 3)
    ).astype(np.float32)
    blob = np.asarray(
        Image.fromarray((blob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    mix = rng.uniform(0.5, 1.0)
    return mix * blob + (1 - mix) * c0


def degrade(img: np.ndarray, rng: random.Random, native_h: int, log: dict | None = None) -> np.ndarray:
    """Camera/screen degradations applied to the composited image only.

    Blur scales with the native text height: a 1.8 px gaussian erases 12 px text
    outright but is realistic camera softness on 40 px text. Det/rec gate what
    reaches the ink model, so training must not contain text they would reject.
    """
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if rng.random() < 0.5:
        sigma = rng.uniform(0.3, min(0.35 + native_h * 0.032, 2.0))
        pil = pil.filter(ImageFilter.GaussianBlur(sigma))
        if log is not None:
            log["blur"] = round(sigma, 2)
    if native_h >= 24 and rng.random() < 0.25:
        # Crude motion blur: directional box kernel. Skipped on tiny text, where even a
        # 3px smear destroys the glyph.
        k = rng.choice([3, 5])
        if log is not None:
            log["motion"] = k
        kernel = [0.0] * (k * k)
        if rng.random() < 0.5:
            for i in range(k):
                kernel[(k // 2) * k + i] = 1.0 / k
        else:
            for i in range(k):
                kernel[i * k + k // 2] = 1.0 / k
        pil = pil.filter(ImageFilter.Kernel((k, k), kernel, scale=1.0))
    if rng.random() < 0.35:
        scale = rng.uniform(0.55, 0.85)
        small = pil.resize(
            (max(8, int(pil.width * scale)), max(8, int(pil.height * scale))), Image.BILINEAR
        )
        pil = small.resize((pil.width, pil.height), Image.BILINEAR)
        if log is not None:
            log["downsample"] = round(scale, 2)
    if rng.random() < 0.7:
        q = rng.randint(45, 95)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
        if log is not None:
            log["jpeg"] = q
    out = np.asarray(pil, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        yy, xx = _coord_grid(h, w)
        angle = rng.uniform(0, 2 * np.pi)
        t = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        shade = rng.uniform(0.55, 1.0) + t * rng.uniform(0.0, 0.45)
        out = out * np.clip(shade, 0.4, 1.2)[..., None]
        if log is not None:
            log["shade"] = 1
    if rng.random() < 0.25:
        # Hard-edged cast shadow: a sharp brightness step across the strip. A strong
        # illumination edge looks like a stroke to the model unless it has trained on
        # shadows that aren't ink (the label is untouched).
        h, w = out.shape[:2]
        yy, xx = _coord_grid(h, w)
        angle = rng.uniform(0, 2 * np.pi)
        proj = (np.cos(angle) * xx / w) + (np.sin(angle) * yy / h)
        edge = rng.uniform(proj.min(), proj.max())
        shadow = np.where(proj < edge, rng.uniform(0.4, 0.8), 1.0).astype(np.float32)
        out = out * shadow[..., None]
        if log is not None:
            log["hardshadow"] = 1
    if rng.random() < 0.6:
        nsig = rng.uniform(0.005, 0.04)
        out = out + np.random.default_rng(rng.getrandbits(32)).normal(
            0, nsig, out.shape
        ).astype(np.float32)
        if log is not None:
            log["noise"] = round(nsig, 3)
    if rng.random() < 0.3:
        lo, hi = rng.uniform(0.0, 0.08), rng.uniform(0.85, 1.0)
        out = out * (hi - lo) + lo
        if log is not None:
            log["squeeze"] = round(hi - lo, 2)
    return np.clip(out, 0, 1)


def legible(img: np.ndarray, cov: np.ndarray, native_h: int) -> bool:
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
    # Small text needs more contrast: its fine inter-stroke gaps vanish at low contrast,
    # so the floor rises as native height shrinks (nh14 ~0.23, nh30+ flat at 0.13).
    thresh = 0.13 + max(0, 30 - native_h) * 0.006
    return float(d.max()) > thresh


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
        cov, fill, native_h, native_w = _render_once(rng, width)
        img, cov_out, ok = _composite_once(rng, cov, fill, native_h, native_w, width)
        if ok:
            break
    return img, cov_out


def stream(rng: random.Random, width: int, reuse: int = 1):
    """Infinite (img, cov) pairs with glyph-raster reuse.

    Rasterizing text dominates generation cost, so each rendered coverage is reused
    across `reuse` composites with fresh background/ink/degradation — dividing the
    rasterization cost by `reuse`. The label repeats within a reuse group (same
    coverage, different appearance), which is benign augmentation for a matte target.
    """
    while True:
        cov, fill, native_h, native_w = _render_once(rng, width)
        for _ in range(reuse):
            img = cov_out = None
            for _try in range(3):
                img, cov_out, ok = _composite_once(rng, cov, fill, native_h, native_w, width)
                if ok:
                    break
            yield img, cov_out


def _render_once(rng: random.Random, width: int, log: dict | None = None):
    # Tail above 48: signage/display text whose native height far exceeds the strip —
    # after the squash its strokes are very thick and glyph interiors are majority-ink,
    # a density regime body text never reaches (validation: station-sign letters
    # mottled when 12–48 was the whole training range).
    if rng.random() < 0.2:
        native_h = int(rng.uniform(48, 160))
    else:
        native_h = int(rng.triangular(12, 48, 20))
    if log is not None:
        log["native_h"] = native_h
    native_w = max(16, round(width * native_h / HEIGHT))
    cov, fill = render_coverage(rng, native_w, native_h, log)
    return cov, fill, native_h, native_w


def _composite_once(rng: random.Random, cov, fill, native_h: int, native_w: int, width: int,
                    log: dict | None = None):
    bg = gradient_field(rng, native_h, native_w, log)
    if rng.random() < 0.15:
        # Drop shadow: an offset dark replica behind the glyphs. It is *not* ink —
        # the label stays `cov` — so the model learns to leave shadows to the
        # background reconstruction instead of matting them.
        dy, dx = rng.randint(1, 3), rng.randint(-2, 3)
        shadow = np.roll(np.roll(cov, dy, axis=0), dx, axis=1)
        bg = bg * (1.0 - (shadow * rng.uniform(0.35, 0.7))[..., None])
        if log is not None:
            log["dropshadow"] = 1
    ink = random_color(rng)
    # Sometimes force low contrast against the background mean.
    if rng.random() < 0.35:
        mean_bg = bg.mean(axis=(0, 1))
        ink = np.clip(mean_bg + np.sign(ink - mean_bg) * rng.uniform(0.05, 0.30), 0, 1)
        if log is not None:
            log["locontrast"] = 1
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
        if log is not None:
            log["outline"] = 1
    img = cov[..., None] * ink_field + (1 - cov[..., None]) * bg
    img = degrade(img, rng, native_h, log)
    ok = legible(img, cov, native_h)
    if log is not None:
        im, bm = cov > 0.6, cov < 0.05
        if im.sum() > 10 and bm.sum() > 10:
            d = np.abs(np.median(img[im], axis=0) - np.median(img[bm], axis=0))
            log["contrast"] = round(float(d.max()), 2)
        log["ok"] = ok
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
    lines = []
    for i in range(args.n):
        # Replicate sample() but capture the per-sample params into `log`. Fresh log
        # per attempt — optional fields would otherwise leak from a failed retry.
        width = rng.choice(WIDTHS)
        img = cov = None
        log: dict = {}
        for _ in range(8):
            log = {}
            c, fl, nh, nw = _render_once(rng, width, log)
            img, cov, ok = _composite_once(rng, c, fl, nh, nw, width, log)
            if ok:
                break
        h, w = cov.shape
        band = 26
        sheet = np.ones((band + h * 2 + 4, max(w, 360), 3), dtype=np.float32)
        sheet[band : band + h, :w] = img
        sheet[band + h + 4 :, :w] = cov[..., None]
        pim = Image.fromarray((sheet * 255).astype(np.uint8))
        deg = " ".join(
            f"{kk}{log[kk]}" for kk in ("blur", "downsample", "jpeg", "noise", "squeeze",
                                        "motion", "shade", "hardshadow") if kk in log)
        head = (f"{i:03d} nh{log.get('native_h')} sz{log.get('size')} k{log.get('thick_k', 0)} "
                f"{'CJK ' if log.get('cjk') else ''}{'HVY ' if log.get('heavy') else ''}"
                f"{log.get('bg', '')} ct{log.get('contrast', '?')}")
        d = ImageDraw.Draw(pim)
        d.text((2, 1), head, fill=(220, 0, 0))
        d.text((2, 13), deg, fill=(0, 0, 200))
        pim.save(os.path.join(args.out, f"sample-{i:03d}.png"))
        lines.append(f"{head} | {deg} | ok={log.get('ok')} font={log.get('font')} "
                     f"stroke{log.get('stroke')} "
                     f"{'locontrast ' if log.get('locontrast') else ''}"
                     f"{'outline ' if log.get('outline') else ''}"
                     f"{'dropshadow' if log.get('dropshadow') else ''}")
    with open(os.path.join(args.out, "params.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.n} annotated samples + params.txt to {args.out}")


if __name__ == "__main__":
    main()

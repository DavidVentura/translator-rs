"""Split Kokoro's all-voices npz into one fp16 npz per voice.

Each output `<voice>.bin` holds a single `<voice>` array of shape (510, 1, 256),
so a voice pack only downloads the voice it offers. Style values are small
and smooth, so fp16 storage loses nothing audible.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", required=True, type=Path, help="upstream voices-v1.0.bin")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("names", nargs="+", help="voice names, e.g. af_heart")
    args = parser.parse_args()

    voices = np.load(args.voices)
    missing = sorted(set(args.names) - set(voices.files))
    if missing:
        raise SystemExit(f"voices not in {args.voices}: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        style = voices[name]
        half = style.astype(np.float16)
        error = float(np.abs(half.astype(np.float32) - style).max())
        out = args.out_dir / f"{name}.bin"
        with out.open("wb") as handle:
            np.savez(handle, **{name: half})
        print(f"wrote {out} {out.stat().st_size} bytes, shape {half.shape}, max fp16 error {error:.2e}")


if __name__ == "__main__":
    main()

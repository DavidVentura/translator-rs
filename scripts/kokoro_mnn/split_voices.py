"""Bundle a subset of Kokoro's all-voices npz into one fp16 npz.

The output holds one `<voice>` array of shape (510, 1, 256) per named voice,
so a language pack downloads only the voices trained on its language. Style
values are small and smooth, so fp16 storage loses nothing audible.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices", required=True, type=Path, help="upstream voices-v1.0.bin")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("names", nargs="+", help="voice names, e.g. ef_dora em_alex")
    args = parser.parse_args()

    voices = np.load(args.voices)
    missing = sorted(set(args.names) - set(voices.files))
    if missing:
        raise SystemExit(f"voices not in {args.voices}: {missing}")

    halves = {}
    for name in args.names:
        style = voices[name]
        halves[name] = style.astype(np.float16)
        error = float(np.abs(halves[name].astype(np.float32) - style).max())
        print(f"{name}: shape {style.shape}, max fp16 error {error:.2e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        np.savez(handle, **halves)
    print(f"wrote {args.out} {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()

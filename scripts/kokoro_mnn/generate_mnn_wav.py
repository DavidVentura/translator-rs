"""Generate a Kokoro WAV from phonemes with the MNN runtime."""

from __future__ import annotations

import argparse
import wave
import zipfile
from io import BytesIO
from pathlib import Path

import MNN.expr as F
import MNN.nn as nn
import numpy as np

from bench import KOKORO_VOCAB


def load_voice_style(voices_path: Path, voice_name: str, token_count: int) -> np.ndarray:
    with zipfile.ZipFile(voices_path) as zf:
        npy_name = f"{voice_name}.npy"
        if npy_name not in zf.namelist():
            names = sorted(n.removesuffix(".npy") for n in zf.namelist() if n.endswith(".npy"))
            raise SystemExit(f"voice {voice_name!r} not found. Available: {', '.join(names)}")
        with zf.open(npy_name) as fh:
            arr = np.load(BytesIO(fh.read()))
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    return arr[min(token_count, arr.shape[0] - 1)].astype(np.float32, copy=False).reshape(1, 256)


def phonemes_to_ids(phonemes: str) -> np.ndarray:
    ids = [0]
    skipped: list[str] = []
    for ch in phonemes:
        if ch in KOKORO_VOCAB:
            ids.append(KOKORO_VOCAB[ch])
        else:
            skipped.append(ch)
    ids.append(0)
    if skipped:
        print(f"warn: skipped {len(skipped)} chars not in Kokoro vocab: {''.join(sorted(set(skipped)))!r}")
    return np.asarray(ids, dtype=np.int32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, normalize: bool) -> None:
    x = audio.astype(np.float32, copy=True)
    if normalize:
        peak = max(0.01, float(np.max(np.abs(x)))) if x.size else 0.01
        x /= peak
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn", required=True)
    parser.add_argument("--voices", default="/home/david/Downloads/voices-v1.0.bin")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--phonemes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    ids = phonemes_to_ids(args.phonemes)
    token_count = ids.size - 2
    style = load_voice_style(Path(args.voices), args.voice, token_count)
    speed = np.asarray([args.speed], dtype=np.float32)

    rt = nn.create_runtime_manager([{"backend": "CPU", "numThread": args.threads}])
    net = nn.load_module_from_file(args.mnn, ["input_ids", "style", "speed"], ["waveform"], runtime_manager=rt)
    inputs = [
        F.const(ids.tobytes(), [1, ids.size], F.NCHW, F.int),
        F.const(style.tobytes(), [1, 256], F.NCHW, F.float),
        F.const(speed.tobytes(), [1], F.NCHW, F.float),
    ]
    audio = np.asarray(F.convert(net.forward(inputs)[0], F.NCHW).read()).copy().reshape(-1)
    write_wav(Path(args.out), audio, 24000, args.normalize)
    print(
        f"wrote {args.out}; voice={args.voice}; tokens={token_count}; "
        f"samples={audio.size}; seconds={audio.size / 24000:.3f}; "
        f"rms={float(np.sqrt(np.mean(audio * audio))):.6f}"
    )


if __name__ == "__main__":
    main()

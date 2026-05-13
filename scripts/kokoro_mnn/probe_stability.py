import argparse
from pathlib import Path

import MNN.expr as F
import MNN.nn as nn
import numpy as np

from bench import load_voice_style, phonemize_to_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn", required=True)
    parser.add_argument("--voices", default="/home/david/Downloads/voices-v1.0.bin")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    ids = np.asarray(phonemize_to_ids(args.text), dtype=np.int32)
    style = load_voice_style(Path(args.voices), len(ids) - 2).reshape(1, 256)
    speed = np.asarray([1.0], dtype=np.float32)

    rt = nn.create_runtime_manager([{"backend": "CPU", "numThread": args.threads}])
    rt.set_cache(".mnn_cache")
    net = nn.load_module_from_file(args.mnn, ["input_ids", "style", "speed"], ["waveform"], runtime_manager=rt)
    inputs = [
        F.const(ids.tobytes(), [1, ids.size], F.NCHW, F.int),
        F.const(style.tobytes(), list(style.shape), F.NCHW, F.float),
        F.const(speed.tobytes(), [1], F.NCHW, F.float),
    ]

    outputs = []
    for _ in range(args.iters):
        out = F.convert(net.forward(inputs)[0], F.NCHW)
        outputs.append(np.asarray(out.read()).copy().reshape(-1).astype(np.float64))

    ref = outputs[0]
    print(f"model={args.mnn} samples={ref.size} rms0={np.sqrt(np.mean(ref * ref)):.6g}")
    for i, out in enumerate(outputs[1:], start=1):
        n = min(ref.size, out.size)
        diff = np.abs(ref[:n] - out[:n])
        rms = float(np.sqrt(np.mean(out[:n] * out[:n])))
        print(f"iter{i}: samples={out.size} mae_vs_0={diff.mean():.6g} max={diff.max():.6g} rms={rms:.6g}")


if __name__ == "__main__":
    main()

"""Locate the Kokoro MNN 4.8/9.6 kHz whistle across generator tensors.

For waveform outputs this reports the same 4.8/9.6 kHz tone prominence used by
whistle_metrics.py. For intermediate tensors it measures narrow FFT energy at
0.2 and 0.4 cycles/sample along the last tensor axis. Those normalized
frequencies correspond to 4.8 and 9.6 kHz in the final 24 kHz waveform and are
also the periodicities expected from a stride-5 / checkerboard-style artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import MNN.expr as F
import MNN.nn as nn
import numpy as np

from bench import load_voice_style, phonemize_to_ids


DEFAULT_TENSORS = [
    "/decoder/decoder/generator/LeakyRelu_output_0",
    "/decoder/decoder/generator/ups.0/ConvTranspose_output_0",
    "/decoder/decoder/generator/Add_3_output_0",
    "/decoder/decoder/generator/ups.1/ConvTranspose_output_0",
    "/decoder/decoder/generator/Add_7_output_0",
    "/decoder/decoder/generator/Div_4_output_0",
    "/decoder/decoder/generator/LeakyRelu_2_output_0",
    "/decoder/decoder/generator/conv_post/Conv_output_0",
    "/decoder/decoder/generator/istft/stft/ConvTranspose_output_0",
    "waveform",
]


def run_model(
    model_path: str,
    output_names: list[str],
    ids: np.ndarray,
    style: np.ndarray,
    speed: np.ndarray,
    threads: int,
) -> list[np.ndarray]:
    config = {"backend": "CPU", "numThread": threads}
    rt = nn.create_runtime_manager([config])
    rt.set_cache(".mnn_cache")
    net = nn.load_module_from_file(
        model_path,
        ["input_ids", "style", "speed"],
        output_names,
        runtime_manager=rt,
    )
    inputs = [
        F.const(ids.tobytes(), [1, ids.size], F.NCHW, F.int),
        F.const(style.tobytes(), list(style.shape), F.NCHW, F.float),
        F.const(speed.tobytes(), [1], F.NCHW, F.float),
    ]
    return [np.asarray(F.convert(out, F.NCHW).read()).copy() for out in net.forward(inputs)]


def db(value: float) -> float:
    return 10.0 * float(np.log10(max(value, 1e-30)))


def normalized_tones(arr: np.ndarray, tones: tuple[float, float] = (0.2, 0.4)) -> dict[str, float]:
    x = arr.astype(np.float64, copy=False)
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    else:
        x = x.reshape(-1, x.shape[-1])

    length = x.shape[-1]
    if length < 32:
        return {f"tone{tone:g}": float("nan") for tone in tones}

    x = x - x.mean(axis=-1, keepdims=True)
    window = np.hanning(length)
    power = np.abs(np.fft.rfft(x * window, axis=-1)) ** 2
    avg_power = power.mean(axis=0) + 1e-30
    freqs = np.fft.rfftfreq(length, d=1.0)

    out: dict[str, float] = {}
    for tone in tones:
        bin_width = 1.0 / length
        tone_half = max(0.0025, 3.0 * bin_width)
        neigh_half = max(0.015, 18.0 * bin_width)
        tone_mask = np.abs(freqs - tone) <= tone_half
        neigh_mask = (np.abs(freqs - tone) > tone_half * 2.0) & (
            np.abs(freqs - tone) <= neigh_half
        )
        tone_power = float(avg_power[tone_mask].mean())
        neigh_power = float(avg_power[neigh_mask].mean())
        total_power = float(avg_power[(freqs > 0.0) & (freqs < 0.5)].mean())
        out[f"tone{tone:g}"] = db(tone_power / neigh_power)
        out[f"rel{tone:g}"] = db(tone_power / total_power)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", default="kokoro-v1.0.patched.i32.while.mnn")
    parser.add_argument("--block128", default="kokoro-v1.0.patched.i32.while.wq8.block128.mnn")
    parser.add_argument("--hqq", default="kokoro-v1.0.patched.i32.while.wq8.hqq.mnn")
    parser.add_argument("--voices", default="/home/david/Downloads/voices-v1.0.bin")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("tensors", nargs="*")
    args = parser.parse_args()

    ids = np.asarray(phonemize_to_ids(args.text), dtype=np.int32)
    style = load_voice_style(Path(args.voices), len(ids) - 2).reshape(1, 256)
    speed = np.asarray([1.0], dtype=np.float32)
    names = args.tensors or DEFAULT_TENSORS
    models = [
        ("fp32", args.fp32),
        ("block128", args.block128),
        ("hqq", args.hqq),
    ]

    outputs = {
        label: run_model(path, names, ids, style, speed, args.threads)
        for label, path in models
    }

    print(
        "tensor                                                   shape                 "
        "b_d0.2 b_d0.4 h_d0.2 h_d0.4 b_rel0.2 b_rel0.4"
    )
    for idx, name in enumerate(names):
        fp = normalized_tones(outputs["fp32"][idx])
        bq = normalized_tones(outputs["block128"][idx])
        hq = normalized_tones(outputs["hqq"][idx])
        shape = "x".join(str(dim) for dim in outputs["fp32"][idx].shape)
        print(
            f"{name[:56]:56s} {shape:20s} "
            f"{bq['tone0.2'] - fp['tone0.2']:+6.2f} "
            f"{bq['tone0.4'] - fp['tone0.4']:+6.2f} "
            f"{hq['tone0.2'] - fp['tone0.2']:+6.2f} "
            f"{hq['tone0.4'] - fp['tone0.4']:+6.2f} "
            f"{bq['rel0.2'] - fp['rel0.2']:+8.2f} "
            f"{bq['rel0.4'] - fp['rel0.4']:+8.2f}"
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import MNN.expr as F
import MNN.nn as nn
import numpy as np

from bench import load_voice_style, phonemize_to_ids


DEFAULT_TENSORS = [
    "/encoder/Cast_output_0",
    "/encoder/Div_output_0",
    "/encoder/MatMul_1_output_0",
    "/decoder/decoder/Concat_output_0",
    "/decoder/decoder/F0_conv/Conv_output_0",
    "/decoder/decoder/N_conv/Conv_output_0",
    "/decoder/decoder/generator/f0_upsamp/Resize_output_0",
    "/decoder/decoder/generator/noise_convs.0/Conv_output_0",
    "/decoder/decoder/generator/ups.0/ConvTranspose_output_0",
    "/decoder/decoder/generator/ups.1/ConvTranspose_output_0",
    "/decoder/decoder/generator/conv_post/Conv_output_0",
    "waveform",
]


def run_model(path, output_names, ids, style, speed, threads):
    rt = nn.create_runtime_manager([{"backend": "CPU", "numThread": threads}])
    rt.set_cache(".mnn_cache")
    net = nn.load_module_from_file(path, ["input_ids", "style", "speed"], output_names, runtime_manager=rt)
    inputs = [
        F.const(ids.tobytes(), [1, ids.size], F.NCHW, F.int),
        F.const(style.tobytes(), list(style.shape), F.NCHW, F.float),
        F.const(speed.tobytes(), [1], F.NCHW, F.float),
    ]
    # read() returns a buffer backed by the MNN variable. Copy before returning so
    # the data survives after the module/runtime objects in this function die.
    return [np.asarray(F.convert(out, F.NCHW).read()).copy() for out in net.forward(inputs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="kokoro-v1.0.patched.i32.mnn")
    parser.add_argument("--b", default="kokoro-v1.0.patched.i32.ptq-ema.mnn")
    parser.add_argument("--voices", default="/home/david/Downloads/voices-v1.0.bin")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("tensors", nargs="*")
    args = parser.parse_args()

    ids = np.asarray(phonemize_to_ids(args.text), dtype=np.int32)
    style = load_voice_style(Path(args.voices), len(ids) - 2).reshape(1, 256)
    speed = np.asarray([1.0], dtype=np.float32)
    names = args.tensors or DEFAULT_TENSORS

    a_out = run_model(args.a, names, ids, style, speed, args.threads)
    b_out = run_model(args.b, names, ids, style, speed, args.threads)
    for name, a, b in zip(names, a_out, b_out):
        af = a.reshape(-1).astype(np.float64)
        bf = b.reshape(-1).astype(np.float64)
        n = min(af.size, bf.size)
        if n:
            diff = np.abs(af[:n] - bf[:n])
            mae = float(diff.mean())
            max_abs = float(diff.max())
            rms_a = float(np.sqrt(np.mean(af[:n] * af[:n])))
            rms_b = float(np.sqrt(np.mean(bf[:n] * bf[:n])))
        else:
            mae = max_abs = rms_a = rms_b = float("nan")
        print(
            f"{name}\n"
            f"  shape_a={a.shape} dtype_a={a.dtype} shape_b={b.shape} dtype_b={b.dtype} "
            f"mae={mae:.6g} max={max_abs:.6g} "
            f"rms_a={rms_a:.6g} rms_b={rms_b:.6g}"
        )


if __name__ == "__main__":
    main()

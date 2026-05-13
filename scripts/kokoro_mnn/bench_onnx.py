"""Bench multiple Kokoro ONNX variants via onnxruntime CPUExecutionProvider."""
import argparse
import subprocess
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bench import KOKORO_VOCAB, phonemize_to_ids, load_voice_style


def bench_model(path, ids, style, threads, warmup, iters):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()
    in_names = [i.name for i in in_meta]
    speed_type = in_meta[2].type
    style_type = in_meta[1].type

    style_arr = style.astype(np.float16 if "float16" in style_type else np.float32)
    if "float" in speed_type and "float16" not in speed_type:
        speed = np.asarray([1.0], dtype=np.float32)
    elif "float16" in speed_type:
        speed = np.asarray([1.0], dtype=np.float16)
    else:
        speed = np.asarray([1], dtype=np.int32)

    feed = {
        in_names[0]: ids.reshape(1, -1),
        in_names[1]: style_arr,
        in_names[2]: speed,
    }

    def run():
        return sess.run(None, feed)[0].reshape(-1)

    for _ in range(warmup):
        run()
    ts = []
    last = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last = run()
        ts.append(time.perf_counter() - t0)
    arr = np.asarray(ts)
    print(
        f"{Path(path).name:>28}: min {arr.min()*1000:7.1f} ms  "
        f"median {np.median(arr)*1000:7.1f} ms  "
        f"mean {arr.mean()*1000:7.1f} ms  "
        f"p95 {np.percentile(arr,95)*1000:7.1f} ms  "
        f"n={iters} samples={last.size} style_dtype={style_arr.dtype} speed_dtype={speed.dtype}"
    )
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--voice", default="en-us")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()

    ids = np.asarray(phonemize_to_ids(args.text, args.voice), dtype=np.int64)
    token_count = ids.size - 2
    style = load_voice_style(Path(args.voices), token_count).reshape(1, 256)
    print(f"tokens={token_count}, threads={args.threads}, warmup={args.warmup}, iters={args.iters}")

    outs = {}
    for p in args.models:
        outs[p] = bench_model(p, ids, style, args.threads, args.warmup, args.iters)

    if len(outs) >= 2:
        names = list(outs)
        ref = outs[names[0]]
        print(f"\nparity vs {Path(names[0]).name}:")
        for n in names[1:]:
            o = outs[n]
            k = min(ref.size, o.size)
            mae = float(np.abs(ref[:k] - o[:k]).mean())
            from numpy.fft import rfft
            sa, sb = np.abs(rfft(ref[:k])), np.abs(rfft(o[:k]))
            sp = float(np.abs(sa - sb).mean() / (sa.mean() + 1e-9))
            print(
                f"  {Path(n).name:>28}: ref_samples={ref.size} this_samples={o.size}; "
                f"MAE={mae:.5f} spectral_rel={sp:.4f}"
            )


if __name__ == "__main__":
    main()

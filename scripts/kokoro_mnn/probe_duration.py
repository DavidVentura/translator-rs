import argparse
import copy
import subprocess
import tempfile
from pathlib import Path

import MNN.expr as F
import MNN.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

from bench import load_voice_style, phonemize_to_ids


def value_info_by_name(model):
    infos = {}
    for info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        infos[info.name] = info
    return infos


def select_outputs(model, every=1, start=0, stop=None, contains=None, op=None):
    selected = []
    for index, node in enumerate(model.graph.node):
        if index < start:
            continue
        if stop is not None and index >= stop:
            continue
        if contains and contains not in node.name:
            continue
        if op and node.op_type != op:
            continue
        if (index - start) % every != 0:
            continue
        for out in node.output:
            if out:
                selected.append((index, node.name, node.op_type, out))
    return selected


def patch_outputs(model, output_names):
    infos = value_info_by_name(model)
    del model.graph.output[:]
    for name in output_names:
        if name not in infos:
            raise SystemExit(f"no value_info for {name}")
        model.graph.output.append(copy.deepcopy(infos[name]))


def run_probe(args):
    model = onnx.load(args.onnx)
    selected = select_outputs(
        model,
        every=args.every,
        start=args.start,
        stop=args.stop,
        contains=args.contains,
        op=args.op,
    )
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no outputs selected")

    output_names = [out for _, _, _, out in selected]
    patch_outputs(model, output_names)

    ids_list = phonemize_to_ids(args.text)
    speed = np.asarray([1.0], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="kokoro_probe_") as tmp:
        onnx_path = Path(tmp) / "probe.onnx"
        mnn_path = Path(tmp) / "probe.mnn"
        onnx.save(model, onnx_path)
        subprocess.run(
            [
                "mnnconvert",
                "-f",
                "ONNX",
                "--modelFile",
                str(onnx_path),
                "--MNNModel",
                str(mnn_path),
                "--useOriginRNNImpl",
            ],
            check=True,
        )

        ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_names = [i.name for i in ort_sess.get_inputs()]
        ids_type = ort_sess.get_inputs()[0].type
        ids_dtype = np.int32 if "int32" in ids_type else np.int64
        ids = np.asarray(ids_list, dtype=ids_dtype)
        style = load_voice_style(Path(args.voices), len(ids) - 2).reshape(1, 256)
        feed = {
            in_names[0]: ids.reshape(1, -1),
            in_names[1]: style,
            in_names[2]: speed,
        }
        ort_outs = ort_sess.run(None, feed)

        rt = nn.create_runtime_manager([{"backend": "CPU", "numThread": args.threads}])
        net = nn.load_module_from_file(str(mnn_path), in_names, output_names, runtime_manager=rt)
        mnn_inputs = [
            F.const(ids.tobytes(), [1, ids.size], F.NCHW, F.int if ids.dtype == np.int32 else F.int64),
            F.const(style.tobytes(), list(style.shape), F.NCHW, F.float),
            F.const(speed.tobytes(), [1], F.NCHW, F.float),
        ]
        mnn_outs = [np.asarray(F.convert(out, F.NCHW).read()) for out in net.forward(mnn_inputs)]

    for meta, ort_arr, mnn_arr in zip(selected, ort_outs, mnn_outs):
        index, node_name, op_type, out_name = meta
        a = np.asarray(ort_arr).reshape(-1)
        b = np.asarray(mnn_arr).reshape(-1)
        n = min(a.size, b.size)
        if n == 0:
            mae = max_abs = float("nan")
        else:
            diff = np.abs(a[:n] - b[:n])
            mae = float(diff.mean())
            max_abs = float(diff.max())
        if mae >= args.mae or max_abs >= args.max_abs or a.shape != b.shape:
            print(
                f"{index:04d} {op_type:20s} {node_name} -> {out_name}\n"
                f"     shape ORT={ort_arr.shape} MNN={mnn_arr.shape} "
                f"mae={mae:.6g} max={max_abs:.6g}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="durations.onnx")
    parser.add_argument("--voices", default="/home/david/Downloads/voices-v1.0.bin")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contains")
    parser.add_argument("--op")
    parser.add_argument("--mae", type=float, default=1e-4)
    parser.add_argument("--max-abs", type=float, default=1e-3)
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()

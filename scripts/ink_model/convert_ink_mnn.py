"""Convert an ink checkpoint to ONNX then MNN (fp16) for on-device inference.

Reads base/levels from the checkpoint so any architecture variant converts.
Dynamic width axis (any strip width); height is fixed at 48.

    python convert_ink_mnn.py --ckpt ckpt/ink-v9-16.pt --out ink.mnn
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

from model import InkUNet

DEFAULT_MNNCONVERT = "/home/david/git/mnn-sys/3rd_party/MNN/build-convert/MNNConvert"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output .mnn path")
    ap.add_argument("--onnx", help="keep the intermediate .onnx here (default: alongside --out)")
    ap.add_argument("--mnnconvert", default=DEFAULT_MNNCONVERT)
    ap.add_argument("--no-fp16", action="store_true", help="emit fp32 weights instead of fp16")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    base, levels = state.get("base", 16), state.get("levels", 2)
    model = InkUNet(base=base, levels=levels)
    model.load_state_dict(state["model"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"ckpt base={base} levels={levels} params={params:,}")

    onnx_path = args.onnx or str(Path(args.out).with_suffix(".onnx"))
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 48, 320),
        onnx_path,
        input_names=["strip"],
        output_names=["ink_logits"],
        dynamic_axes={"strip": {3: "width"}, "ink_logits": {3: "width"}},
        opset_version=17,
    )

    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for width in (160, 320, 504):
        x = torch.rand(1, 3, 48, width)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"strip": x.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        assert diff < 1e-3, f"onnx mismatch at width {width}: {diff:.2e}"
    print(f"onnx verified (max diff < 1e-3) -> {onnx_path}")

    cmd = [args.mnnconvert, "-f", "ONNX", "--modelFile", onnx_path,
           "--MNNModel", args.out, "--bizCode", "ink"]
    if not args.no_fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)
    print(f"-> {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

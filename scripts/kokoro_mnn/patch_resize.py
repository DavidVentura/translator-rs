"""Patch Kokoro fp32 ONNX into the stable MNN conversion source.

The output is `kokoro-v1.0.patched.i32.dft.onnx` by default. It applies the
Resize/Round cleanup and normalizes the token-id graph input to
`input_ids: INT32` so MNN's Gather path matches ORT. Kokoro ONNX exports have
used both `tokens` and `input_ids`; this script accepts either source name.

For each Resize node listed in TARGETS we:
  - clear the `scales` input (index 2)
  - feed an explicit `sizes` input (index 3) computed from the input's runtime
    Shape, multiplied/divided by the integer factor.

This removes any rounding-mode dependence between ONNX runtimes.

It also rewrites three generator pieces that are slow in MNN or overflow in
fp16: the harmonic-source STFT becomes a fixed DFT convolution, the decomposed
instance norms become InstanceNormalization, and the phase atan2 no longer
divides by an exact zero.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort


# (node name, integer factor; positive=upsample by N, negative=downsample by |N|)
TARGETS = [
    ("/encoder/F0.1/upsample/Resize", 2),
    ("/encoder/N.1/upsample/Resize", 2),
    ("/decoder/decoder/generator/f0_upsamp/Resize", 300),
    ("/decoder/decoder/generator/m_source/l_sin_gen/Resize", -300),
    ("/decoder/decoder/generator/m_source/l_sin_gen/Resize_1", 300),
    ("/decoder/decoder/decode.3/upsample/Resize", 2),
]


def replace_round_with_floor_half_up(graph: gs.Graph) -> None:
    """ONNX Round = banker's rounding (round half to even); MNN's Round
    implementation rounds half away from zero. They disagree on x.5 values.
    Replace Round with Floor(x + 0.5), which both runtimes evaluate identically.
    """
    for node in list(graph.nodes):
        if node.op != "Round":
            continue
        x = node.inputs[0]
        unique = node.name.replace("/", "_").lstrip("_")
        half = gs.Constant(f"{unique}/half", values=np.array(0.5, dtype=np.float32))
        added = gs.Variable(f"{unique}/x_plus_half", dtype=np.float32)
        add_node = gs.Node(op="Add", inputs=[x, half], outputs=[added])
        # Rewire the downstream output through Floor instead of Round.
        original_out = node.outputs[0]
        floor_node = gs.Node(op="Floor", name=f"{node.name}_as_floor", inputs=[added], outputs=[original_out])
        node.outputs = []  # detach so cleanup() removes it
        graph.nodes.extend([add_node, floor_node])
        print(f"replaced {node.name} (Round) with Add+Floor")


def normalize_token_input_to_int32(graph: gs.Graph) -> None:
    """MNN 3.5 miscomputes Gather when the ONNX indices input is INT64.
    Kokoro token ids fit in INT32, and ONNX Gather accepts INT32 indices, so
    narrowing this model input keeps ORT semantics while avoiding the MNN bug.
    Normalize the public name to `input_ids` because the app integration uses
    named MNN inputs.
    """
    for inp in graph.inputs:
        if inp.name in ("input_ids", "tokens"):
            old_name = inp.name
            inp.name = "input_ids"
            inp.dtype = np.int32
            print(f"changed token input {old_name} to input_ids INT32")
            return
    names = ", ".join(inp.name for inp in graph.inputs)
    raise SystemExit(f"token input not found; expected input_ids or tokens, got: {names}")


# Large enough to stay a normal number in fp16, small enough to be far below
# any bin that carries signal.
PHASE_EPS = 1e-4


def constant_value(tensor: gs.Tensor) -> np.ndarray:
    if isinstance(tensor, gs.Constant):
        return np.asarray(tensor.values)
    producer = tensor.inputs[0]
    if producer.op != "Constant":
        raise SystemExit(f"{tensor.name} is not a constant (produced by {producer.op})")
    return np.asarray(producer.attrs["value"].values)


def consumers(tensor: gs.Tensor, op: str) -> list[gs.Node]:
    # A node that reads a tensor twice (x * x) is listed twice in tensor.outputs.
    unique = {id(n): n for n in tensor.outputs}
    return [n for n in unique.values() if n.op == op]


def replace_stft_with_dft_conv(graph: gs.Graph) -> None:
    """MNN's Stft kernel is single-threaded and slow for this n_fft=20 STFT, and
    its fp16 path returns exact zeros that the exported atan2 turns into NaN. A
    Conv1d over the windowed DFT basis computes the same frames.

    torch.stft gives the DC and Nyquist bins an exact +0 imaginary part, so
    torch.angle is +pi when their real part is negative. The exported atan2
    only takes the +pi branch for a strictly positive imaginary part, so those
    two channels get a PHASE_EPS bias to land on the branch Kokoro was trained on.
    """
    stfts = [n for n in graph.nodes if n.op == "STFT"]
    if len(stfts) != 1:
        raise SystemExit(f"expected one STFT node, found {len(stfts)}")
    stft = stfts[0]
    signal, frame_step, window, frame_length = stft.inputs
    hop = int(constant_value(frame_step))
    n_fft = int(constant_value(frame_length))
    win = constant_value(window).astype(np.float64)
    if stft.attrs.get("onesided", 1) != 1 or len(win) != n_fft:
        raise SystemExit(f"unexpected STFT configuration on {stft.name}")

    bins = n_fft // 2 + 1
    n = np.arange(n_fft)
    basis = np.zeros((2 * bins, 1, n_fft))
    for k in range(bins):
        basis[2 * k, 0] = win * np.cos(2 * np.pi * k * n / n_fft)
        basis[2 * k + 1, 0] = -win * np.sin(2 * np.pi * k * n / n_fft)
    nyquist_imag = 2 * (bins - 1) + 1
    basis[1, 0] = 0
    basis[nyquist_imag, 0] = 0
    bias = np.zeros(2 * bins, np.float32)
    bias[1] = bias[nyquist_imag] = PHASE_EPS

    name = stft.name
    signal_3d = gs.Variable(f"{name}/signal_3d", dtype=np.float32)
    conv_out = gs.Variable(f"{name}/conv", dtype=np.float32)
    bins_split = gs.Variable(f"{name}/bins_split", dtype=np.float32)
    out = stft.outputs[0]
    stft.outputs = []
    graph.nodes.extend([
        gs.Node(op="Unsqueeze", inputs=[signal, gs.Constant(f"{name}/axis1", np.array([1], np.int64))], outputs=[signal_3d]),
        gs.Node(
            op="Conv",
            name=f"{name}/dft",
            attrs={"strides": [hop], "kernel_shape": [n_fft]},
            inputs=[signal_3d, gs.Constant(f"{name}/basis", basis.astype(np.float32)), gs.Constant(f"{name}/bias", bias)],
            outputs=[conv_out],
        ),
        gs.Node(op="Reshape", inputs=[conv_out, gs.Constant(f"{name}/split_shape", np.array([0, bins, 2, -1], np.int64))], outputs=[bins_split]),
        gs.Node(op="Transpose", attrs={"perm": [0, 3, 1, 2]}, inputs=[bins_split], outputs=[out]),
    ])
    print(f"replaced {name} (STFT n_fft={n_fft} hop={hop}) with a DFT Conv")


def guard_phase_division(graph: gs.Graph) -> None:
    """The phase is exported as atan(imag / real); in silent frames fp16 gives
    real == imag == 0 and the 0/0 NaN spreads through every instance norm into
    the whole utterance."""
    atans = [n for n in graph.nodes if n.op == "Atan"]
    if len(atans) != 1:
        raise SystemExit(f"expected one Atan node, found {len(atans)}")
    div = atans[0].inputs[0].inputs[0]
    if div.op != "Div":
        raise SystemExit(f"Atan input comes from {div.op}, expected Div")
    real = div.inputs[1]
    unique = div.name.replace("/", "_").lstrip("_")
    is_zero = gs.Variable(f"{unique}/real_is_zero", dtype=np.bool_)
    guarded = gs.Variable(f"{unique}/real_guarded", dtype=np.float32)
    graph.nodes.extend([
        gs.Node(op="Equal", inputs=[real, gs.Constant(f"{unique}/zero", np.array(0, np.float32))], outputs=[is_zero]),
        gs.Node(op="Where", inputs=[is_zero, gs.Constant(f"{unique}/eps", np.array(PHASE_EPS, np.float32)), real], outputs=[guarded]),
    ])
    div.inputs[1] = guarded
    print(f"guarded {div.name} against a zero real part")


def find_instance_norms(graph: gs.Graph) -> list[tuple[gs.Tensor, gs.Node, float]]:
    """AdaIN exports InstanceNorm1d as (x - mean) / sqrt(mean((x - mean)^2) + eps)
    over the time axis: seven elementwise/reduction passes per norm."""
    found = []
    for mean in [n for n in graph.nodes if n.op == "ReduceMean"]:
        x = mean.inputs[0]
        subs = [s for s in consumers(mean.outputs[0], "Sub") if s.inputs[0] is x]
        if len(subs) != 1:
            continue
        centered = subs[0].outputs[0]
        squares = [m for m in consumers(centered, "Mul") if m.inputs[0] is m.inputs[1]]
        if len(squares) != 1:
            continue
        variances = consumers(squares[0].outputs[0], "ReduceMean")
        if len(variances) != 1:
            continue
        eps_adds = consumers(variances[0].outputs[0], "Add")
        if len(eps_adds) != 1:
            continue
        sqrts = consumers(eps_adds[0].outputs[0], "Sqrt")
        if len(sqrts) != 1:
            continue
        divs = [d for d in consumers(centered, "Div") if d.inputs[1] is sqrts[0].outputs[0]]
        if len(divs) != 1:
            continue
        axes = constant_value(mean.inputs[1]).reshape(-1).tolist() if len(mean.inputs) > 1 else mean.attrs["axes"]
        if axes not in ([2], [-1]):
            raise SystemExit(f"{mean.name} normalizes over axes {axes}, expected the time axis")
        eps = float(constant_value(eps_adds[0].inputs[1]).reshape(-1)[0])
        found.append((x, divs[0], eps))
    return found


def channel_counts(graph: gs.Graph, tensors: list[gs.Tensor]) -> list[int]:
    """Most normalized tensors have data-dependent shapes that ONNX shape
    inference cannot resolve, so read the channel axis from one real run."""
    probe = graph.copy()
    by_name = probe.tensors()
    probe.outputs = [by_name[t.name] for t in tensors]
    for out in probe.outputs:
        out.dtype = np.float32
    session = ort.InferenceSession(gs.export_onnx(probe).SerializeToString(), providers=["CPUExecutionProvider"])
    ids = np.full((1, 20), 50, np.int32)
    ids[0, 0] = ids[0, -1] = 0
    outs = session.run(None, {"input_ids": ids, "style": np.zeros((1, 256), np.float32), "speed": np.ones(1, np.float32)})
    return [o.shape[1] for o in outs]


def fuse_instance_norms(graph: gs.Graph) -> None:
    norms = find_instance_norms(graph)
    if not norms:
        raise SystemExit("no decomposed instance norm found")
    channels = channel_counts(graph, [x for x, _, _ in norms])
    for (x, div, eps), c in zip(norms, channels):
        out = div.outputs[0]
        div.outputs = []
        graph.nodes.append(gs.Node(
            op="InstanceNormalization",
            name=f"{div.name}_instance_norm",
            attrs={"epsilon": eps},
            inputs=[x, gs.Constant(f"{div.name}/in_scale", np.ones(c, np.float32)), gs.Constant(f"{div.name}/in_bias", np.zeros(c, np.float32))],
            outputs=[out],
        ))
    print(f"fused {len(norms)} decomposed instance norms into InstanceNormalization")


def patch(input_path: Path, output_path: Path) -> None:
    graph = gs.import_onnx(onnx.load(str(input_path)))
    node_by_name = {n.name: n for n in graph.nodes}

    normalize_token_input_to_int32(graph)
    replace_round_with_floor_half_up(graph)

    for name, factor in TARGETS:
        node = node_by_name.get(name)
        if node is None:
            raise SystemExit(f"node not found: {name}")
        if node.op != "Resize":
            raise SystemExit(f"node {name} is not Resize: {node.op}")
        if len(node.inputs) < 3:
            raise SystemExit(f"node {name} has fewer than 3 inputs")

        src = node.inputs[0]
        unique = name.replace("/", "_").lstrip("_")

        # Shape(src) -> [N, C, T]
        shape_out = gs.Variable(f"{unique}/shape_out", dtype=np.int64)
        shape_node = gs.Node(op="Shape", inputs=[src], outputs=[shape_out])

        # Slice the time dim: shape[2:3]
        starts = gs.Constant(f"{unique}/slice_starts", values=np.array([2], dtype=np.int64))
        ends = gs.Constant(f"{unique}/slice_ends", values=np.array([3], dtype=np.int64))
        axes = gs.Constant(f"{unique}/slice_axes", values=np.array([0], dtype=np.int64))
        t_dim = gs.Variable(f"{unique}/t_dim", dtype=np.int64)
        slice_node = gs.Node(op="Slice", inputs=[shape_out, starts, ends, axes], outputs=[t_dim])

        if factor > 0:
            scale_const = gs.Constant(f"{unique}/factor", values=np.array([factor], dtype=np.int64))
            t_new = gs.Variable(f"{unique}/t_new", dtype=np.int64)
            arith = gs.Node(op="Mul", inputs=[t_dim, scale_const], outputs=[t_new])
        else:
            scale_const = gs.Constant(f"{unique}/factor", values=np.array([-factor], dtype=np.int64))
            t_new = gs.Variable(f"{unique}/t_new", dtype=np.int64)
            arith = gs.Node(op="Div", inputs=[t_dim, scale_const], outputs=[t_new])

        # Take N, C dims (shape[0:2])
        nc_starts = gs.Constant(f"{unique}/nc_starts", values=np.array([0], dtype=np.int64))
        nc_ends = gs.Constant(f"{unique}/nc_ends", values=np.array([2], dtype=np.int64))
        nc_axes = gs.Constant(f"{unique}/nc_axes", values=np.array([0], dtype=np.int64))
        nc_dim = gs.Variable(f"{unique}/nc_dim", dtype=np.int64)
        nc_slice = gs.Node(op="Slice", inputs=[shape_out, nc_starts, nc_ends, nc_axes], outputs=[nc_dim])

        sizes_out = gs.Variable(f"{unique}/sizes", dtype=np.int64)
        concat_node = gs.Node(op="Concat", attrs={"axis": 0}, inputs=[nc_dim, t_new], outputs=[sizes_out])

        graph.nodes.extend([shape_node, slice_node, arith, nc_slice, concat_node])

        # Resize signature: X, roi (opt), scales (opt), sizes (opt).
        # Clear scales (index 2), set sizes (index 3).
        empty_in = gs.Variable.empty()
        node.inputs[2] = empty_in
        if len(node.inputs) >= 4:
            node.inputs[3] = sizes_out
        else:
            node.inputs.append(sizes_out)

        print(f"patched {name}: factor={factor}")

    replace_stft_with_dft_conv(graph)
    guard_phase_division(graph)
    graph.cleanup().toposort()
    fuse_instance_norms(graph)
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), str(output_path))
    print(f"wrote {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="kokoro-v1.0.onnx")
    ap.add_argument("--output", default="kokoro-v1.0.patched.i32.dft.onnx")
    args = ap.parse_args()
    patch(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

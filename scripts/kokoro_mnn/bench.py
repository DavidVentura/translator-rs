"""Compare Kokoro v1.0 inference: onnxruntime vs MNN.

Loads voice style from voices-v1.0.bin (NPZ), feeds the same phoneme id sequence
through ORT and MNN, prints per-run latency and output audio MAE.
"""

import argparse
import subprocess
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort

import MNN
import MNN.expr as F
import MNN.nn as nn


# Kokoro v1.0 phoneme vocabulary (mirrors piper-rs/src/kokoro.rs kokoro_vocab()).
KOKORO_VOCAB = {
    ";":1,":":2,",":3,".":4,"!":5,"?":6,"—":9,"…":10,"\"":11,"(":12,")":13,
    "“":14,"”":15," ":16,"̃":17,"ʣ":18,"ʥ":19,"ʦ":20,"ʨ":21,"ᵝ":22,
    "ꭧ":23,"A":24,"I":25,"O":31,"Q":33,"S":35,"T":36,"W":39,"Y":41,"ᵊ":42,
    "a":43,"b":44,"c":45,"d":46,"e":47,"f":48,"h":50,"i":51,"j":52,"k":53,"l":54,
    "m":55,"n":56,"o":57,"p":58,"q":59,"r":60,"s":61,"t":62,"u":63,"v":64,"w":65,
    "x":66,"y":67,"z":68,"ɑ":69,"ɐ":70,"ɒ":71,"æ":72,"β":75,
    "ɔ":76,"ɕ":77,"ç":78,"ɖ":80,"ð":81,"ʤ":82,"ə":83,
    "ɚ":85,"ɛ":86,"ɜ":87,"ɟ":90,"ɡ":92,"ɥ":99,
    "ɨ":101,"ɪ":102,"ʝ":103,"ɯ":110,"ɰ":111,"ŋ":112,
    "ɳ":113,"ɲ":114,"ɴ":115,"ø":116,"ɸ":118,"θ":119,
    "œ":120,"ɹ":123,"ɾ":125,"ɻ":126,"ʁ":128,"ɽ":129,
    "ʂ":130,"ʃ":131,"ʈ":132,"ʧ":133,"ʊ":135,"ʋ":136,
    "ʌ":138,"ɣ":139,"ɤ":140,"χ":142,"ʎ":143,"ʒ":147,
    "ʔ":148,"ˈ":156,"ˌ":157,"ː":158,"ʰ":162,"ʲ":164,
    "↓":169,"→":171,"↗":172,"↘":173,"ᵻ":177,
}


def phonemize_to_ids(text: str, espeak_voice: str = "en-us") -> list[int]:
    """Call espeak-ng to get IPA, then map to Kokoro vocab ids (0-padded both ends)."""
    out = subprocess.run(
        ["espeak-ng", "-v", espeak_voice, "--ipa", "-q", text],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ids = [0]
    skipped = []
    for ch in out:
        if ch in KOKORO_VOCAB:
            ids.append(KOKORO_VOCAB[ch])
        else:
            skipped.append(ch)
    ids.append(0)
    if skipped:
        print(f"warn: dropped {len(skipped)} chars not in vocab: {''.join(set(skipped))!r}")
    print(f"phonemes: {out!r} -> {len(ids)-2} tokens")
    return ids


def load_voice_style(voices_path: Path, token_count: int) -> np.ndarray:
    """Replicate piper-rs's voice selection: pick row min(token_count, last) from a [511, 1, 256] array."""
    with zipfile.ZipFile(voices_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".npy"))
        if not names:
            raise SystemExit(f"no .npy entries in {voices_path}")
        with z.open(names[0]) as fh:
            arr = np.load(BytesIO(fh.read()))
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    idx = min(token_count, arr.shape[0] - 1)
    return arr[idx].astype(np.float32, copy=False)  # [256]


def bench(label, run_once, warmup, iters):
    for _ in range(warmup):
        run_once()
    times = []
    last_out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last_out = run_once()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    print(
        f"{label:>14}: median {arr.min()*1000:7.1f} ms  "
        f"mean {arr.mean()*1000:7.1f} ms  "
        f"p95 {np.percentile(arr, 95)*1000:7.1f} ms  "
        f"n={iters} (samples={last_out.shape[-1]})"
    )
    return last_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mnn", required=True)
    ap.add_argument("--voices", required=True)
    # Native MNN LSTM models mutate recurrent state across forwards, so the
    # default keeps parity/audio checks on the first inference.
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--voice", default="en-us")
    ap.add_argument("--out-prefix", default="out")
    ap.add_argument("--mnn-precision", choices=["normal", "high", "low", "lowBF"], default="normal")
    ap.add_argument("--mnn-memory", choices=["normal", "high", "low"], default="normal")
    ap.add_argument("--mnn-power", choices=["normal", "high", "low"], default="normal")
    args = ap.parse_args()

    ids_list = phonemize_to_ids(args.text, args.voice)
    token_count = len(ids_list) - 2
    style = load_voice_style(Path(args.voices), token_count).reshape(1, 256)
    # fp32 export uses float speed; int8 export uses int speed.
    speed_f = np.asarray([1.0], dtype=np.float32)
    speed_i = np.asarray([1], dtype=np.int32)

    print(
        f"phoneme tokens (excl. sentinels): {token_count}; "
        f"style shape {style.shape}; threads {args.threads}"
    )

    # ---- ORT ----
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    in_meta = ort_sess.get_inputs()
    in_names = [i.name for i in in_meta]
    out_names = [o.name for o in ort_sess.get_outputs()]
    ids_type = in_meta[0].type
    ids_dtype = np.int32 if "int32" in ids_type else np.int64
    ids = np.asarray(ids_list, dtype=ids_dtype)
    speed_type = in_meta[2].type
    speed = speed_f if "float" in speed_type else speed_i
    print(f"ORT inputs: {in_names}; ids type {ids_type}; speed type {speed_type}; outputs: {out_names}")

    feed = {
        in_names[0]: ids.reshape(1, -1),
        in_names[1]: style,
        in_names[2]: speed,
    }

    def run_ort():
        return ort_sess.run(None, feed)[0].reshape(-1)

    ort_audio = bench("onnxruntime", run_ort, args.warmup, args.iters)

    # ---- MNN ----
    mnn_config = {"backend": "CPU", "numThread": args.threads}
    if args.mnn_precision != "normal":
        mnn_config["precision"] = args.mnn_precision
    if args.mnn_memory != "normal":
        mnn_config["memory"] = args.mnn_memory
    if args.mnn_power != "normal":
        mnn_config["power"] = args.mnn_power
    print(f"MNN config: {mnn_config}")
    rt = nn.create_runtime_manager([mnn_config])
    rt.set_cache(".mnn_cache")
    net = nn.load_module_from_file(args.mnn, in_names, out_names, runtime_manager=rt)

    ids_mnn_type = F.int if ids.dtype == np.int32 else F.int64
    ids_v = F.const(ids.tobytes(), [1, ids.size], F.NCHW, ids_mnn_type)
    style_v = F.const(style.tobytes(), list(style.shape), F.NCHW, F.float)
    speed_dtype = F.float if speed.dtype == np.float32 else F.int
    speed_v = F.const(speed.tobytes(), [speed.size], F.NCHW, speed_dtype)

    def run_mnn():
        out = net.forward([ids_v, style_v, speed_v])
        # Ensure NCHW layout before reading raw buffer.
        v = F.convert(out[0], F.NCHW)
        return np.asarray(v.read()).copy().reshape(-1)

    mnn_audio = bench("MNN", run_mnn, args.warmup, args.iters)

    # ---- Parity ----
    n = min(ort_audio.size, mnn_audio.size)
    a, b = ort_audio[:n], mnn_audio[:n]
    mae = float(np.abs(a - b).mean())
    rms_a = float(np.sqrt((a * a).mean()))
    rms_b = float(np.sqrt((b * b).mean()))
    # Spectral L1 distance, less sensitive to phase
    from numpy.fft import rfft
    sa = np.abs(rfft(a))
    sb = np.abs(rfft(b))
    spectral_mae = float(np.abs(sa - sb).mean() / (sa.mean() + 1e-9))
    print(
        f"parity: ORT samples={ort_audio.size} MNN samples={mnn_audio.size}; "
        f"compared first {n}; MAE={mae:.5f}; RMS_ort={rms_a:.4f} RMS_mnn={rms_b:.4f}; "
        f"spectral_mae_rel={spectral_mae:.4f}"
    )

    # Dump wavs for ear inspection
    import wave
    def write_wav(path, x):
        x16 = np.clip(x, -1, 1)
        x16 = (x16 * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(x16.tobytes())
    ort_wav = f"{args.out_prefix}_ort.wav"
    mnn_wav = f"{args.out_prefix}_mnn.wav"
    write_wav(ort_wav, ort_audio)
    write_wav(mnn_wav, mnn_audio)
    print(f"wrote {ort_wav} and {mnn_wav}")


if __name__ == "__main__":
    main()

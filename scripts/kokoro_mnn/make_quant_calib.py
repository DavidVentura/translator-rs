import argparse
import json
import shutil
from pathlib import Path

import MNN.expr as F
import MNN.nn as nn
import numpy as np

from bench import load_voice_style, phonemize_to_ids


TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "A small test sentence checks timing and pronunciation.",
    "Numbers like one two three should not change the rhythm.",
    "Short words and longer phrases exercise the duration model.",
    "This calibration sample is intentionally plain English.",
]


def write_txt(path: Path, values: np.ndarray) -> None:
    with path.open("w") as fh:
        for value in values.reshape(-1):
            fh.write(f"{float(value):.9g}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn", default="kokoro-v1.0.patched.i32.mnn")
    parser.add_argument("--voices", default="/home/david/Downloads/voices-v1.0.bin")
    parser.add_argument("--out", default="/tmp/kokoro-mnn-quant-calib")
    parser.add_argument("--config-out")
    parser.add_argument("--limit", type=int, default=len(TEXTS))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    input_names = ["input_ids", "style", "speed"]
    output_names = ["waveform"]
    rt = nn.create_runtime_manager([{"backend": "CPU", "numThread": args.threads}])
    net = nn.load_module_from_file(args.mnn, input_names, output_names, runtime_manager=rt)

    for index, text in enumerate(TEXTS[: args.limit]):
        sample_dir = out_dir / f"input_{index}"
        sample_dir.mkdir()

        ids = np.asarray(phonemize_to_ids(text), dtype=np.int32)
        style = load_voice_style(Path(args.voices), len(ids) - 2).reshape(1, 256)
        speed = np.asarray([1.0], dtype=np.float32)

        inputs = [
            F.const(ids.tobytes(), [1, ids.size], F.NCHW, F.int),
            F.const(style.tobytes(), list(style.shape), F.NCHW, F.float),
            F.const(speed.tobytes(), [1], F.NCHW, F.float),
        ]
        waveform = np.asarray(F.convert(net.forward(inputs)[0], F.NCHW).read())

        write_txt(sample_dir / "input_ids.txt", ids)
        write_txt(sample_dir / "style.txt", style)
        write_txt(sample_dir / "speed.txt", speed)
        write_txt(sample_dir / "waveform.txt", waveform)

        config = {
            "inputs": [
                {"name": "input_ids", "shape": [1, int(ids.size)]},
                {"name": "style", "shape": [1, 256]},
                {"name": "speed", "shape": [1]},
            ],
            "outputs": output_names,
        }
        (sample_dir / "input.json").write_text(json.dumps(config, indent=4))

    config_path = Path(args.config_out) if args.config_out else out_dir.parent / f"{out_dir.name}.json"
    quant_config = {
        "path": str(out_dir),
        "used_sample_num": min(args.limit, len(TEXTS)),
        "feature_quantize_method": "EMA",
        "weight_quantize_method": "MAX_ABS",
        "feature_clamp_value": 127,
        "weight_clamp_value": 127,
        "batch_size": 1,
        "quant_bits": 8,
        "skip_quant_op_names": [],
        "input_type": "sequence",
        "debug": False,
    }
    config_path.write_text(json.dumps(quant_config, indent=4))
    print(config_path)


if __name__ == "__main__":
    main()

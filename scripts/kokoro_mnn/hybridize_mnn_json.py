"""Create MNN JSON hybrids by restoring selected ops from an fp32 MNN JSON.

This is a debugging helper for Kokoro weight-only quantization. It keeps the
quantized graph topology and tensor indexes intact, but replaces selected op
parameters (`main`) with the matching fp32 op parameters by name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="quantized/base MNN JSON")
    parser.add_argument("--fp32", required=True, help="fp32 MNN JSON")
    parser.add_argument("--out", required=True, help="output hybrid MNN JSON")
    parser.add_argument("--op", action="append", required=True, help="op name to restore from fp32")
    args = parser.parse_args()

    base_path = Path(args.base)
    fp32_path = Path(args.fp32)
    out_path = Path(args.out)

    base = json.loads(base_path.read_text())
    fp32 = json.loads(fp32_path.read_text())
    fp32_by_name = {op.get("name"): op for op in fp32["oplists"]}

    restored: list[str] = []
    for op in base["oplists"]:
        name = op.get("name")
        if name not in args.op:
            continue
        src = fp32_by_name.get(name)
        if src is None:
            raise SystemExit(f"fp32 JSON does not contain op {name!r}")
        if op.get("type") != src.get("type") or op.get("main_type") != src.get("main_type"):
            raise SystemExit(
                f"op kind mismatch for {name!r}: "
                f"base=({op.get('type')}, {op.get('main_type')}) "
                f"fp32=({src.get('type')}, {src.get('main_type')})"
            )
        op["main"] = src["main"]
        restored.append(name)

    missing = sorted(set(args.op) - set(restored))
    if missing:
        raise SystemExit(f"base JSON does not contain ops: {missing}")

    out_path.write_text(json.dumps(base, separators=(",", ":")))
    print(f"wrote {out_path} with {len(restored)} restored op(s)")
    for name in restored:
        print(f"  {name}")


if __name__ == "__main__":
    main()

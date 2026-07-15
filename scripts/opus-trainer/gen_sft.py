#!/usr/bin/env python3
"""Resumable SFT-pair generator via `claude -p` (no API key needed).

Reads a source corpus (JSON list of strings, or list of {"text": ...}),
translates it in batches through the Claude CLI, and writes one file per
batch so it can be stopped and resumed at any time. Re-running skips batches
already completed and validated.

Example:
  gen_sft.py --src en_article.json --model opus \
      --from English --to Uyghur --script "Perso-Arabic, Xinjiang usage" \
      --out out/article_en2ug --batch 100

Outputs under --out:
  batches/batch_00000.json ...   per-batch target lists (resume unit)
  targets.json                   all targets concatenated, in order
  pairs.jsonl                    {"src":..., "tgt":...} one per line
  cost.log                       running USD-equivalent per batch
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

SYS = "You are a professional machine translation engine. Follow the output format exactly and emit nothing else."


def load_sources(p: str) -> list[str]:
    raw = json.load(open(p, encoding="utf-8"))
    if raw and isinstance(raw[0], dict):
        return [r["text"] for r in raw]
    return list(raw)


def strip_fences(t: str) -> str:
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def call_claude(model: str, prompt: str) -> dict:
    cmd = [
        "claude", "-p", "--model", model, "--output-format", "json",
        "--allowedTools", "", "--exclude-dynamic-system-prompt-sections",
        "--system-prompt", SYS, prompt,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"claude exited {r.returncode}: {r.stderr[:400]}")
    return json.loads(r.stdout)


def translate_batch(model: str, src_lang: str, tgt_lang: str, script: str, srcs: list[str]) -> tuple[list[str], float]:
    instr = (
        f"Translate each {src_lang} sentence in this JSON array into natural, fluent "
        f"{tgt_lang} ({script}). Output ONLY a JSON array of exactly {len(srcs)} "
        f"{tgt_lang} strings, in the same order, one per input. No commentary, no "
        f"romanization, no markdown fences.\n\n" + json.dumps(srcs, ensure_ascii=False)
    )
    last_err = ""
    for attempt in range(2):
        resp = call_claude(model, instr if attempt == 0 else instr + "\n\nReturn a bare JSON array only.")
        cost = float(resp.get("total_cost_usd", 0.0))
        try:
            out = json.loads(strip_fences(resp["result"]))
        except json.JSONDecodeError as e:
            last_err = f"parse: {e}"
            continue
        if not isinstance(out, list) or len(out) != len(srcs):
            last_err = f"count: got {len(out) if isinstance(out, list) else type(out)} want {len(srcs)}"
            continue
        return [str(x) for x in out], cost
    raise RuntimeError(f"batch failed after retries ({last_err})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", required=True, help="opus | sonnet | haiku | full id")
    ap.add_argument("--from", dest="src_lang", required=True)
    ap.add_argument("--to", dest="tgt_lang", required=True)
    ap.add_argument("--script", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="cap total sources (0 = all)")
    a = ap.parse_args()

    srcs = load_sources(a.src)
    if a.limit:
        srcs = srcs[:a.limit]
    out = pathlib.Path(a.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(exist_ok=True)

    total_cost = 0.0
    n = len(srcs)
    for start in range(0, n, a.batch):
        chunk = srcs[start:start + a.batch]
        bf = out / "batches" / f"batch_{start:06d}.json"
        if bf.exists():
            try:
                if len(json.load(open(bf))) == len(chunk):
                    print(f"[skip] {start}-{start+len(chunk)} done", flush=True)
                    continue
            except Exception:
                pass
        try:
            tgts, cost = translate_batch(a.model, a.src_lang, a.tgt_lang, a.script, chunk)
        except Exception as e:
            (out / "failures" / f"batch_{start:06d}.txt").write_text(str(e))
            print(f"[FAIL] {start}: {e}", file=sys.stderr, flush=True)
            continue
        json.dump(tgts, open(bf, "w"), ensure_ascii=False)
        total_cost += cost
        with open(out / "cost.log", "a") as f:
            f.write(f"{start}\t{len(chunk)}\t{cost:.4f}\t{total_cost:.4f}\n")
        print(f"[ok] {start}-{start+len(chunk)}  cost={cost:.3f}  cum={total_cost:.3f}", flush=True)

    # assemble in order; only if fully complete
    targets, pairs, complete = [], [], True
    for start in range(0, n, a.batch):
        bf = out / "batches" / f"batch_{start:06d}.json"
        if not bf.exists():
            complete = False
            break
        chunk = srcs[start:start + a.batch]
        tg = json.load(open(bf))
        targets += tg
        pairs += [{"src": s, "tgt": t} for s, t in zip(chunk, tg)]
    if complete:
        json.dump(targets, open(out / "targets.json", "w"), ensure_ascii=False)
        with open(out / "pairs.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[DONE] {len(pairs)} pairs -> {out}/pairs.jsonl  total_cost={total_cost:.3f}", flush=True)
    else:
        print(f"[PARTIAL] some batches missing; rerun to continue. total_cost={total_cost:.3f}", flush=True)


if __name__ == "__main__":
    main()

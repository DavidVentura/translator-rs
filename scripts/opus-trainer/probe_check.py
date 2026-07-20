#!/usr/bin/env python3
"""Score a probe decode by reference-free mechanical checks.

Manual reading found what chrF/COMET could not (2026-07-20: COMET ranked an
inverted safety instruction above its own median and a correct line 6th-worst),
but reading does not survive the next model. The mechanical part of what reading
caught is checked here; the rest still needs a human, and this does not pretend
otherwise.

All checks are derived from the source line, so they need no reference, no
annotation and no target-language knowledge — they work on a new pair the day it
exists:

- numbers must survive: "140 over 90" -> "140" loses a vital, "500 mg" a dose
- length blowup: output/source word ratio
- repetition: a repeated n-gram, e.g. "hallo" -> "hello in the hello"
- copy-through (output == source) and empty output

What this deliberately does NOT catch: meaning inversions (discontinue ->
continue) and wrong word choice, both of which are fluent and short. Those stay a
reading job.

    probe_check.py probes.jsonl HYP_FILE <en2tl|tl2en>
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

NUM = re.compile(r"\d+(?:[.,]\d+)?")

# Spelled-out forms, so a digit rendered as a word is not reported as dropped.
WORD_NUM = {
    "0": ("zero", "sero"), "1": ("one", "isa", "isang"), "2": ("two", "dalawa", "dalawang"),
    "3": ("three", "tatlo", "tatlong"), "4": ("four", "apat"),
    "5": ("five", "lima", "limang"), "6": ("six", "anim"),
    "7": ("seven", "pito", "pitong"), "8": ("eight", "walo", "walong"),
    "9": ("nine", "siyam"), "10": ("ten", "sampu", "sampung"),
    "15": ("fifteen", "labinlima"), "24": ("twenty-four", "dalawampu't apat"),
    "45": ("forty-five", "apatnapu't lima"),
    "302": ("three hundred two",), "350": ("three hundred and fifty",),
}


@dataclass(frozen=True)
class Failure:
    kind: str
    detail: str


def check_numbers(src: str, hyp: str) -> list[Failure]:
    low = hyp.lower()
    lost = [n for n in NUM.findall(src)
            if n not in hyp and not any(w in low for w in WORD_NUM.get(n, ()))]
    if not lost:
        return []
    return [Failure("number_dropped", f"{lost} absent from output")]


def check_degenerate(src: str, hyp: str, max_ratio: float) -> list[Failure]:
    if not hyp.strip():
        return [Failure("empty", "empty output")]
    out: list[Failure] = []
    s_words, h_words = src.split(), hyp.split()
    # A floor of a few words, or legitimate expansion of a one-word sign
    # ("Push" -> "Itulak ang pinto") reads as a blowup.
    if len(h_words) > max(3, len(s_words) * max_ratio):
        out.append(Failure("length_blowup",
                           f"{len(s_words)} src words -> {len(h_words)} hyp words"))
    lowered = [w.lower().strip(".,!?;:\"'“”") for w in h_words]
    # Unigram repeats only for very short sources: "hallo" -> "hello in the hello"
    # is degenerate, but a repeated word in a long sentence is usually "the".
    sizes = (3, 2, 1) if len(s_words) <= 3 else (3, 2)
    for n in sizes:
        grams = [tuple(lowered[i:i + n]) for i in range(len(lowered) - n + 1)]
        if n == 1:
            grams = [g for g in grams if len(g[0]) > 2]
        # 3+ occurrences, not 2: rhetorical parallelism is normal and correct
        # ("Walang gagalaw, walang masasaktan" -> "No one will move, no one will
        # get hurt"), while real degeneracy repeats a span over and over
        # ("the All-compassionate, the All-compassionate, the All-compassionate").
        repeats = {g for g in grams if grams.count(g) >= 3 and any(t for t in g)}
        if repeats:
            out.append(Failure("repetition", f"repeated {n}-gram {sorted(repeats)[0]}"))
            break
    if hyp.strip().lower() == src.strip().lower():
        out.append(Failure("copy_through", "output identical to source"))
    return out


def check(src: str, hyp: str, max_ratio: float = 3.0) -> list[Failure]:
    """Every reference-free check for one line. The reusable entry point."""
    return check_numbers(src, hyp) + check_degenerate(src, hyp, max_ratio)


def load_sources(path: Path, direction: str) -> list[tuple[str, str, float]]:
    """(source, category, max_ratio) from either a probes .jsonl or a plain .txt.

    Plain text is what the shared adversarial set is: source lines only, no
    references and no categories, because it must work for a pair whose target
    language nobody on the team writes.
    """
    if path.suffix != ".jsonl":
        return [(l, "-", 3.0) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
    src_key = "en" if direction == "en2tl" else "tl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        out.append((p[src_key], p["category"], p.get("max_ratio", 3.0)))
    return out


def main() -> None:
    probes_p, hyp_p, direction = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    probes = load_sources(probes_p, direction)
    hyps = hyp_p.read_text(encoding="utf-8").splitlines()
    if len(probes) != len(hyps):
        sys.exit(f"line-count mismatch: {len(probes)} probes vs {len(hyps)} hyps")

    by_cat: dict[str, list[int]] = {}
    failures: list[tuple[int, str, str, str, list[Failure]]] = []

    for i, ((src, category, max_ratio), hyp) in enumerate(zip(probes, hyps)):
        found = check(src, hyp, max_ratio)
        by_cat.setdefault(category, []).append(0 if found else 1)
        if found:
            failures.append((i, category, src, hyp, found))

    n = sum(len(v) for v in by_cat.values())
    total = sum(sum(v) for v in by_cat.values())
    print(f"=== {Path(hyp_p).name} [{direction}]  CLEAN {total}/{n} ({100*total/n:.1f}%)")
    for cat in sorted(by_cat):
        v = by_cat[cat]
        print(f"  {cat:12s} {sum(v):3d}/{len(v):3d}")
    print()
    for i, cat, src, hyp, found in failures:
        print(f"[{i:3d}] {cat}\n      SRC {src}\n      HYP {hyp}")
        for f in found:
            print(f"      !!  {f.kind}: {f.detail}")


if __name__ == "__main__":
    main()

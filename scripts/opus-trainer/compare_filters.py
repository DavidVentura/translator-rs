#!/usr/bin/env python3
"""Prove a filter change leaves other pairs untouched.

Loads prep_data.py + registers.py from two directories (for example a
`git show <base>:...` export and the working tree), runs both versions'
clean_pair() and register extras over the same raw OPUS zips, and counts pairs
whose result differs. The Khmer changes are gated to unspaced-script lines; on
en-ka, en-tl and en-ug TED2020 + Tatoeba (46,528 pairs) this reported 0 diffs.

    mkdir /tmp/base && for f in prep_data.py registers.py; do
        git show master:scripts/opus-trainer/$f > /tmp/base/$f; done
    compare_filters.py /tmp/base scripts/opus-trainer <dir of *.en-xx.zip>

LID is not compared: it is a fixed fastText model the change does not touch.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import zipfile


def load(directory: str):
    sys.path.insert(0, directory)
    for name in ("registers", "prep_data"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("prep_data", f"{directory}/prep_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reg = sys.modules["registers"]
    sys.path.remove(directory)
    return mod, reg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir")
    ap.add_argument("new_dir")
    ap.add_argument("raw_dir", type=pathlib.Path)
    a = ap.parse_args()
    args = argparse.Namespace(min_tokens=1, max_tokens=150, max_chars=1000, ratio=3.0, max_word_len=150,
                              word_ratio=0.6, alpha_ratio=0.4, num_ratio=1.0)
    base, breg = load(a.base_dir)
    new, nreg = load(a.new_dir)
    any_diff = False
    for zp in sorted(a.raw_dir.glob("*.zip")):
        corpus, pair = zp.name.split(".")[0], zp.name.split(".")[1]
        lang = pair.split("-")[1]
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            S = z.read(next(n for n in names if n.endswith(".en"))).decode("utf-8", "replace").splitlines()
            T = z.read(next(n for n in names if n.endswith("." + lang))).decode("utf-8", "replace").splitlines()
        reg_b = breg.REGISTER.get(corpus, breg.Register.HUMAN)
        reg_n = nreg.REGISTER.get(corpus, nreg.Register.HUMAN)
        diff = 0
        for s, t in zip(S, T):
            x = base.clean_pair(s, t, args)
            y = new.clean_pair(s, t, args)
            # ui_strip too: the UI word check changed, and it only runs on that register.
            x = x and breg.apply_extra(reg_b, *x) and breg.ui_strip(*x)
            y = y and nreg.apply_extra(reg_n, *y) and nreg.ui_strip(*y)
            diff += x != y
        any_diff |= bool(diff)
        print(f"{zp.name}: {len(S)} pairs, {diff} differ")
    sys.exit(1 if any_diff else 0)


if __name__ == "__main__":
    main()

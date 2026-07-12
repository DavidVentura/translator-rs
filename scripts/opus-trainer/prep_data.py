#!/usr/bin/env python3
"""Phase 1: build cleaned en<->X parallel bitext + joint SentencePiece vocab.

Resolves OPUS moses resources for the pair (via `opus_get -l`), downloads an
allowlist of corpora (localization junk and entity-only sets are skipped),
streams each zip through cleaning (Moses de-escape, subtitle/markup artifact
strip, wiki-boilerplate fix, alpha/word-ratio + num/url mismatch filters ported
from mozilla/translations' OpusCleaner, length/ratio filters, and fastText
language ID on both sides), deduplicates on disk with `sort -u`, and trains a
joint unigram SPM. Designed to run on a scratch box: raw shards are deleted
after cleaning and everything is gzip-streamed, so peak disk stays small.

Outputs under <workdir>:
    clean/train.<pair>.<src>.gz  clean/train.<pair>.<tgt>.gz
    spm/vocab.<pair>.spm (+ .vocab)

Setup (on the box):
    python3 -m venv venv
    ./venv/bin/pip install opustools sentencepiece fasttext-wheel "numpy<2"
    curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
import sys
import unicodedata
import zipfile
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import fasttext

# corpora worth keeping for MT (skip Ubuntu/GNOME/translatewiki/ELRC*: localization
# junk; XLEnt: entity pairs; ParaCrawl-Bonus: dup of ParaCrawl)
ALLOWLIST = {
    "NLLB", "CCAligned", "CCMatrix", "OpenSubtitles", "ParaCrawl", "WikiMatrix",
    "wikimedia", "bible-uedin", "TED2020", "tico-19", "Tatoeba", "QED",
}
URL_RE = re.compile(r"(https://\S+/OPUS-([^/]+)/\S+/moses/\S+\.txt\.zip)")


def resolve_urls(venv_bin: Path, src: str, tgt: str) -> list[tuple[str, str]]:
    out = subprocess.run(
        [str(venv_bin / "opus_get"), "-s", src, "-t", tgt, "-p", "moses", "-l"],
        capture_output=True, text=True,
    )
    seen: dict[str, str] = {}
    for line in out.stdout.splitlines():
        m = URL_RE.search(line)
        if not m:
            continue
        url, name = m.group(1), m.group(2)
        if name in ALLOWLIST and name not in seen:
            seen[name] = url
    return list(seen.items())


def members(zf: zipfile.ZipFile, ext: str) -> str:
    hits = [n for n in zf.namelist() if n.endswith("." + ext)]
    if len(hits) != 1:
        raise ValueError(f"expected one *.{ext} member, got {hits}")
    return hits[0]


# Subtitle/markup artifacts (OpenSubtitles ♪ lyric markers, [Applause]/[Music]
# annotations, <i> tags) leak into the student and get emitted at inference
# ("Kumusta kayo" -> "♪ How are you ♪"). Strip them from the data here rather
# than filtering the model output at runtime.
ARTIFACTS = re.compile(r"[♩♪♫♬♭♮♯]|\[[^\]]{0,40}\]|</?[a-zA-Z][^>]*>")


def strip_artifacts(s: str) -> str:
    return re.sub(r"\s+", " ", ARTIFACTS.sub(" ", s)).strip()


# The filters below are ported from mozilla/translations' OpusCleaner chain
# (hplt-project/OpusCleaner). Mined OPUS data is noisier than we were treating
# it: escaped markup, wiki boilerplate, mostly-symbol junk lines, and pairs
# whose numbers/URLs don't line up (a misalignment tell). Cleaning the corpus
# is orthogonal to teacher quality — the teacher only fixes the target side.

# Moses special-char de-escaping. &amp; must be last so "&amp;lt;" un-escapes to
# "&lt;" rather than "<".
DEESCAPE = [
    ("&bar;", "|"), ("&#124;", "|"), ("&lt;", "<"), ("&gt;", ">"),
    ("&bra;", "["), ("&ket;", "]"), ("&quot;", '"'), ("&apos;", "'"),
    ("&#91;", "["), ("&#93;", "]"), ("&#39;", "'"), ("&nbsp;", " "),
    ("&amp;", "&"),
]
FOOTNOTE = re.compile(r"\[[0-9]+\]")
WIKILINK = re.compile(r"\[\[(?:.+?\|)?(.+?)\]\]")
HEADING = re.compile(r"(==+)(.+?)\1")
WIKI_CODE = re.compile(r"\.mw-parser-output")
URL = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b"
    r"[-a-zA-Z0-9()@:%_+.~#?&/=]*"
)
NUM_EXPR = re.compile(r"""
    (?:
        (?P<sign>(?:(?<=\s)|^)[-+])   # leading sign only when detached from a word
        |\b
    )
    (?:0*)                            # ignore leading zeros when comparing
    (?P<value>\d+(?:[.,:]\d+)*)
    \b
""", re.X)


def deescape(s: str) -> str:
    for esc, ch in DEESCAPE:
        s = s.replace(esc, ch)
    return s


def fix_wiki(s: str) -> str:
    s = FOOTNOTE.sub("", s)
    s = WIKILINK.sub(r"\1", s)
    return HEADING.sub("", s)


def alpha_ratio_ok(s: str, word_ratio: float, alpha_ratio: float) -> bool:
    toks = s.split()
    if not toks:
        return False
    lettered = sum(1 for t in toks if any(unicodedata.category(c)[0] == "L" for c in t))
    if lettered / len(toks) < word_ratio:
        return False
    nonspace = len(s) - s.count(" ")
    letters = sum(1 for c in s if unicodedata.category(c)[0] == "L")
    return letters / nonspace >= alpha_ratio


def _nums(s: str) -> set[str]:
    return {(m["sign"] or "") + re.sub(r"[^\d]+", "*", m["value"]) for m in NUM_EXPR.finditer(s)}


def num_mismatch_ok(s: str, t: str, ratio: float) -> bool:
    a, b = _nums(s), _nums(t)
    if not a and not b:
        return True
    return (len(a & b) + 1) / (len(a ^ b) + 1) >= ratio


def url_mismatch_ok(s: str, t: str) -> bool:
    return set(URL.findall(s)) == set(URL.findall(t))


def max_word_ok(s: str, t: str, limit: int) -> bool:
    return not any(len(tok) > limit for field in (s, t) for tok in field.split(" "))


def clean_pair(s_raw: str, t_raw: str, args) -> tuple[str, str] | None:
    """All non-langid filtering + transforms. Returns the cleaned pair, or None
    to drop it. Pure and picklable-input, so it runs in worker processes."""
    s = deescape(s_raw.replace("\t", " "))
    t = deescape(t_raw.replace("\t", " "))
    if WIKI_CODE.search(s) or WIKI_CODE.search(t):
        return None
    s = strip_artifacts(fix_wiki(s))
    t = strip_artifacts(fix_wiki(t))
    if not s or not t or s == t:
        return None
    if not max_word_ok(s, t, args.max_word_len):
        return None
    sw, tw = s.split(), t.split()
    if not (args.min_tokens <= len(sw) <= args.max_tokens):
        return None
    if not (args.min_tokens <= len(tw) <= args.max_tokens):
        return None
    if max(len(sw), len(tw)) / min(len(sw), len(tw)) > args.ratio:
        return None
    if max(len(s), len(t)) > args.max_chars:
        return None
    if not alpha_ratio_ok(s, args.word_ratio, args.alpha_ratio):
        return None
    if not alpha_ratio_ok(t, args.word_ratio, args.alpha_ratio):
        return None
    if not num_mismatch_ok(s, t, args.num_ratio):
        return None
    if not url_mismatch_ok(s, t):
        return None
    return s, t


# One giant corpus (NLLB is ~6x all others combined) dominates the clean, so we
# shard by LINE across a worker pool, not by corpus. Each worker loads its own
# fastText model; the parent only reads/decompresses zips and writes survivors.
_LID = None
_WORKER_ARGS = None


def _init_worker(lid_model: str, args) -> None:
    global _LID, _WORKER_ARGS
    _LID = fasttext.load_model(lid_model)
    _WORKER_ARGS = args


def _clean_batch(task: tuple[str, list]) -> tuple[str, int, list[str]]:
    name, rows = task
    args = _WORKER_ARGS
    src_buf: list[str] = []
    tgt_buf: list[str] = []
    for bs, bt in rows:
        pair = clean_pair(bs.decode("utf-8", "replace"), bt.decode("utf-8", "replace"), args)
        if pair:
            src_buf.append(pair[0])
            tgt_buf.append(pair[1])
    kept: list[str] = []
    if src_buf:
        sl, sp = _LID.predict(src_buf)
        tl, tp = _LID.predict(tgt_buf)
        want_s, want_t = "__label__" + args.src, "__label__" + args.lang
        for s, t, slab, spr, tlab, tpr in zip(src_buf, tgt_buf, sl, sp, tl, tp):
            if slab[0] == want_s and tlab[0] == want_t and spr[0] >= args.min_lid and tpr[0] >= args.min_lid:
                kept.append(f"{s}\t{t}")
    return name, len(rows), kept


def read_batches(urls, raw: Path, pair: str, src: str, tgt: str, skip_download: bool, batch: int = 20000):
    for name, url in urls:
        zp = raw / f"{name}.{pair}.zip"
        if not (skip_download and zp.exists()):
            print(f"[{name}] downloading {url}", file=sys.stderr)
            subprocess.run(["curl", "-sL", "--fail", "-o", str(zp), url], check=True)
        with zipfile.ZipFile(zp) as zf:
            sf, tf = members(zf, src), members(zf, tgt)
            with zf.open(sf) as fs, zf.open(tf) as ft:
                rows: list = []
                for bs, bt in zip(fs, ft):
                    rows.append((bs, bt))
                    if len(rows) >= batch:
                        yield name, rows
                        rows = []
                if rows:
                    yield name, rows
        zp.unlink()  # reclaim disk immediately


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="tl", help="the non-English side (OPUS code)")
    ap.add_argument("--src", default="en")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--lid-model", default="lid.176.ftz")
    ap.add_argument("--only", default="", help="comma list of corpora to restrict to (smoke test)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--max-word-len", type=int, default=150)
    ap.add_argument("--word-ratio", type=float, default=0.6, help="min fraction of tokens that contain a letter")
    ap.add_argument("--alpha-ratio", type=float, default=0.4, help="min fraction of non-space chars that are letters")
    ap.add_argument("--num-ratio", type=float, default=1.0, help="min numbers-overlap/mismatch ratio between sides")
    ap.add_argument("--min-lid", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="clean worker processes")
    ap.add_argument("--skip-spm", action="store_true", help="skip joint SPM training (reuse an existing vocab)")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--sort-mem", default="4G")
    args = ap.parse_args()

    src, tgt = args.src, args.lang
    pair = f"{src}{tgt}"
    wd = Path(args.workdir).resolve()
    raw, clean, spm, tmp = wd / "raw", wd / "clean", wd / "spm", wd / "tmp"
    for d in (raw, clean, spm, tmp):
        d.mkdir(parents=True, exist_ok=True)
    venv_bin = Path(sys.executable).parent

    urls = resolve_urls(venv_bin, src, tgt)
    if args.only:
        keep = set(args.only.split(","))
        urls = [(n, u) for n, u in urls if n in keep]
    if not urls:
        sys.exit(f"no OPUS moses corpora resolved for {src}-{tgt}")
    print(f"corpora for {src}-{tgt}: {', '.join(n for n, _ in urls)}", file=sys.stderr)

    all_tsv = clean / f"all.{pair}.tsv"
    kept_by: dict[str, int] = {n: 0 for n, _ in urls}
    seen_by: dict[str, int] = {n: 0 for n, _ in urls}
    batches = read_batches(urls, raw, pair, src, tgt, args.skip_download)
    max_inflight = args.jobs * 4  # bound queued batches so a huge corpus can't blow up RAM

    def collect(sink, inflight, drain_all: bool) -> None:
        while inflight and (drain_all or len(inflight) >= max_inflight):
            name, seen, kept = inflight.popleft().get()
            seen_by[name] += seen
            kept_by[name] += len(kept)
            for line in kept:
                sink.write(line + "\n")
            print(f"  cleaned {sum(seen_by.values())}", end="\r", file=sys.stderr)

    with open(all_tsv, "w", encoding="utf-8") as sink, \
            Pool(args.jobs, initializer=_init_worker, initargs=(args.lid_model, args)) as pool:
        inflight: deque = deque()
        for task in batches:
            inflight.append(pool.apply_async(_clean_batch, (task,)))
            collect(sink, inflight, drain_all=False)
        collect(sink, inflight, drain_all=True)
    for name in kept_by:
        print(f"[{name}] kept {kept_by[name]}/{seen_by[name]}", file=sys.stderr)
    total_kept, total_seen = sum(kept_by.values()), sum(seen_by.values())

    dedup_tsv = clean / f"dedup.{pair}.tsv"
    print(f"dedup {total_kept} pairs -> sort -u", file=sys.stderr)
    env = {**os.environ, "LC_ALL": "C"}
    with open(dedup_tsv, "w") as out:
        subprocess.run(
            ["sort", "-u", "-S", args.sort_mem, "--parallel", str(os.cpu_count() or 4),
             "-T", str(tmp), str(all_tsv)],
            check=True, stdout=out, env=env,
        )
    all_tsv.unlink()

    src_gz = clean / f"train.{pair}.{src}.gz"
    tgt_gz = clean / f"train.{pair}.{tgt}.gz"
    n_final = 0
    with open(dedup_tsv, encoding="utf-8") as f, \
            gzip.open(src_gz, "wt", encoding="utf-8") as fs, \
            gzip.open(tgt_gz, "wt", encoding="utf-8") as ft:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            fs.write(parts[0] + "\n")
            ft.write(parts[1] + "\n")
            n_final += 1
    dedup_tsv.unlink()

    if args.skip_spm:
        print(f"\nDONE {pair}: {n_final} pairs (from {total_seen} raw), SPM skipped", file=sys.stderr)
        print(f"  {src_gz}\n  {tgt_gz}", file=sys.stderr)
        return

    print(f"training joint SPM ({args.vocab_size}) on {n_final} pairs", file=sys.stderr)
    import sentencepiece as spmlib
    spm_prefix = spm / f"vocab.{pair}"
    corpus_txt = tmp / f"spm_input.{pair}.txt"
    with open(corpus_txt, "w", encoding="utf-8") as w, \
            gzip.open(src_gz, "rt", encoding="utf-8") as fs, \
            gzip.open(tgt_gz, "rt", encoding="utf-8") as ft:
        for line in fs:
            w.write(line)
        for line in ft:
            w.write(line)
    spmlib.SentencePieceTrainer.train(
        input=str(corpus_txt), model_prefix=str(spm_prefix), vocab_size=args.vocab_size,
        model_type="unigram", character_coverage=1.0, byte_fallback=True,
        input_sentence_size=10_000_000, shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4, bos_id=-1, eos_id=0, unk_id=1, pad_id=-1,
    )
    corpus_txt.unlink()

    print(f"\nDONE {pair}: {n_final} pairs (from {total_seen} raw)", file=sys.stderr)
    print(f"  {src_gz}\n  {tgt_gz}\n  {spm_prefix}.model", file=sys.stderr)


if __name__ == "__main__":
    main()

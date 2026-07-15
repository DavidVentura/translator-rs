from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from .ledger import Ledger
from .store import Store
from .step import Run
from .types import Kind, Name, RunId

SCRIPTS = Path(__file__).resolve().parents[2] / "opus-trainer"


def _emit(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _open(run: str, root: Path) -> Run:
    rid = RunId(run)
    store = Store(root)
    store.run_dir(rid).mkdir(parents=True, exist_ok=True)
    return Run(id=rid, store=store, ledger=Ledger(store.ledger_path(rid)), scripts=SCRIPTS)


def cmd_put(a: argparse.Namespace) -> int:
    r = _open(a.run, a.root)
    art = r.store.put(r.id, Name(a.name), Path(a.path), Kind(a.kind))
    r.ledger.register(art)
    _emit({"ok": True, "artifact": art.to_json()})
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    r = _open(a.run, a.root)
    _emit(
        {
            "run": a.run,
            "steps": [
                {
                    "key": s.key,
                    "step": s.step,
                    "status": s.status.value,
                    "seconds": round(s.finished - s.started, 1) if s.finished else None,
                    "log": str(s.log),
                    "outputs": sorted(s.outputs),
                }
                for s in r.ledger.steps.values()
            ],
            "artifacts": {k: v.to_json() for k, v in r.ledger.artifacts.items()},
        }
    )
    return 0


def cmd_log(a: argparse.Namespace) -> int:
    r = _open(a.run, a.root)
    matches = [s for s in r.ledger.steps.values() if s.step == a.step]
    if not matches:
        _emit({"ok": False, "error": f"no step {a.step} in run {a.run}"})
        return 1
    path = matches[-1].log
    text = path.read_text() if path.is_file() else ""
    lines = text.splitlines()
    if a.grep:
        pat = re.compile(a.grep)
        lines = [ln for ln in lines if pat.search(ln)]
    if a.tail:
        lines = lines[-a.tail :]
    _emit({"ok": True, "log": str(path), "matched": len(lines), "lines": lines})
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    flow_path = Path(a.flow).resolve()
    spec = importlib.util.spec_from_file_location("flow", flow_path)
    if spec is None or spec.loader is None:
        _emit({"ok": False, "error": f"cannot load flow {flow_path}"})
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = _open(a.run, a.root)
    result = mod.main(r, a.rest)
    _emit({"ok": True, "run": a.run, "result": result})
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="pipe")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("put", help="promote a file into the store under a name")
    sp.add_argument("name")
    sp.add_argument("path")
    sp.add_argument("--kind", choices=[k.value for k in Kind], default=Kind.LINES.value)
    sp.set_defaults(fn=cmd_put)

    sp = sub.add_parser("status", help="ledger view of the run")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("log", help="read a step's log from disk")
    sp.add_argument("step")
    sp.add_argument("--grep")
    sp.add_argument("--tail", type=int)
    sp.set_defaults(fn=cmd_log)

    sp = sub.add_parser("run", help="run a flow script, skipping completed steps")
    sp.add_argument("flow")
    sp.add_argument("rest", nargs="*")
    sp.set_defaults(fn=cmd_run)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

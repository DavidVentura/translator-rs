from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import sys
from pathlib import Path

from .config import Config
from .ledger import Ledger
from .reaper import reap
from .registry import ImageRef
from .ssh import SshHost
from .store import Store
from .step import Run
from .types import Kind, Name, RunId
from .vast import VastApi, VastError



def _emit(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _open(run: str, cfg: Config) -> Run:
    rid = RunId(run)
    store = Store(cfg.root)
    store.run_dir(rid).mkdir(parents=True, exist_ok=True)
    return Run(id=rid, store=store, ledger=Ledger(store.ledger_path(rid)), scripts=cfg.scripts)


def cmd_put(a: argparse.Namespace) -> int:
    r = _open(a.run, a.cfg)
    art = r.store.put(r.id, Name(a.name), Path(a.path), Kind(a.kind))
    r.ledger.register(art)
    _emit({"ok": True, "artifact": art.to_json()})
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    r = _open(a.run, a.cfg)
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
    r = _open(a.run, a.cfg)
    matches = [s for s in r.ledger.steps.values() if s.step == a.step]
    if not matches:
        _emit({"ok": False, "error": f"no step {a.step} in run {a.run}"})
        return 1
    rec = matches[-1]
    text = ""
    source = str(rec.log)
    if rec.log.is_file():
        text = rec.log.read_text()
    elif rec.job is not None and rec.job.ssh is not None:
        # Still on a rented box: read it live rather than waiting for the step to finish.
        h = SshHost(
            host=rec.job.ssh.host, port=rec.job.ssh.port, user=rec.job.ssh.user, key=a.cfg.vast_key
        )
        text = h.read_file(rec.job.dir / "log") or ""
        source = f"{rec.job.host}:{rec.job.dir / 'log'}"
        h.close()
    lines = text.splitlines()
    if a.grep:
        pat = re.compile(a.grep)
        lines = [ln for ln in lines if pat.search(ln)]
    if a.tail:
        lines = lines[-a.tail :]
    _emit({"ok": True, "log": source, "matched": len(lines), "lines": lines})
    return 0


FLOWS = Path(__file__).resolve().parents[1] / "flows"


def cmd_run(a: argparse.Namespace) -> int:
    flow_path = FLOWS / f"{a.flow}.py"
    if not flow_path.is_file():
        known = sorted(f.stem for f in FLOWS.glob("*.py") if f.stem != "__init__")
        _emit({"ok": False, "error": f"no flow {a.flow!r}", "known": known})
        return 1
    spec = importlib.util.spec_from_file_location("flow", flow_path)
    if spec is None or spec.loader is None:
        _emit({"ok": False, "error": f"cannot load flow {flow_path}"})
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = _open(a.run, a.cfg)
    pidfile = r.store.run_dir(r.id) / "flow.pid"
    pidfile.write_text(str(os.getpid()))
    try:
        result = mod.main(r, a.rest)
    finally:
        pidfile.unlink(missing_ok=True)
    _emit({"ok": True, "run": a.run, "result": result})
    return 0


def cmd_abort(a: argparse.Namespace) -> int:
    """Tear down a run's live activity: flow process, rented boxes, leases.

    Ledger and artifacts stay — completed steps remain memoized, and the next
    `pipe run` resumes from exactly where the aborted one left off.
    """
    r = _open(a.run, a.cfg)
    killed = None
    pidfile = r.store.run_dir(r.id) / "flow.pid"
    if pidfile.is_file():
        pid = int(pidfile.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            killed = pid
        except ProcessLookupError:
            pass
        pidfile.unlink(missing_ok=True)
    v = VastApi(Store(a.cfg.root).leases_dir())
    torn_down = []
    for uid, lease in v.leases().items():
        if lease.run != a.run:
            continue
        try:
            v.destroy(lease.instance_id)
            torn_down.append({"instance": lease.instance_id, "step": lease.step})
        except VastError as e:
            torn_down.append({"instance": lease.instance_id, "step": lease.step, "error": str(e)})
        v.drop_lease(uid)
    _emit({"ok": True, "run": a.run, "flow_pid_killed": killed, "instances": torn_down})
    return 0


def cmd_offers(a: argparse.Namespace) -> int:
    v = VastApi(Store(a.cfg.root).leases_dir())
    offers = v.search(
        gpu_name=a.gpu,
        min_reliability=a.min_reliability,
        min_inet_down=a.min_inet_down,
        disk_gb=a.disk,
        max_dph=a.max_dph,
    )
    _emit(
        {
            "matched": len(offers),
            "offers": [
                {
                    "id": o.id,
                    "gpu": o.gpu_name,
                    "dph": o.dph_total,
                    "reliability": round(o.reliability, 4),
                    "inet_down": o.inet_down,
                    "disk": o.disk_space,
                    "where": o.geolocation,
                }
                for o in offers[: a.limit]
            ],
        }
    )
    return 0


def cmd_version(a: argparse.Namespace) -> int:
    _emit({"synced": a.cfg.synced_at(), "root": str(a.cfg.root), "scripts": str(a.cfg.scripts)})
    return 0


def cmd_image(a: argparse.Namespace) -> int:
    _emit({"image": a.image, "digest": ImageRef.parse(a.image).digest()})
    return 0


def cmd_boxes(a: argparse.Namespace) -> int:
    v = VastApi(Store(a.cfg.root).leases_dir())
    leases = v.leases()
    rows = []
    for i in v.instances():
        uid = i.label.removeprefix("pipe:") if i.label and i.label.startswith("pipe:") else None
        rows.append(
            {
                "id": i.id,
                "status": i.status,
                "gpu": i.gpu_name,
                "label": i.label,
                "ssh": f"{i.ssh_host}:{i.ssh_port}" if i.ssh_host else None,
                "lease": leases[uid].to_json() if uid and uid in leases else None,
            }
        )
    _emit({"instances": rows})
    return 0


def cmd_destroy(a: argparse.Namespace) -> int:
    v = VastApi(Store(a.cfg.root).leases_dir())
    v.destroy(a.instance_id)
    for uid, lease in v.leases().items():
        if lease.instance_id == a.instance_id:
            v.drop_lease(uid)
    _emit({"ok": True, "destroyed": a.instance_id})
    return 0


def cmd_reap(a: argparse.Namespace) -> int:
    v = VastApi(Store(a.cfg.root).leases_dir())
    actions = reap(v, dry_run=a.dry_run)
    _emit(
        {
            "ok": True,
            "dry_run": a.dry_run,
            "actions": [
                {"instance": r.instance_id, "reason": r.reason, "destroyed": r.destroyed}
                for r in actions
            ],
        }
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="pipe")
    p.add_argument("--run", default="adhoc")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("put", help="promote a file into the store under a name")
    sp.add_argument("name")
    sp.add_argument("path")
    sp.add_argument("--kind", choices=[k.value for k in Kind], default=Kind.LINES.value)
    sp.set_defaults(fn=cmd_put)

    sp = sub.add_parser("status", help="ledger view of the run")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("log", help="read a step's log; live from the box if still running")
    sp.add_argument("step")
    sp.add_argument("--grep")
    sp.add_argument("--tail", type=int)
    sp.set_defaults(fn=cmd_log)

    sp = sub.add_parser("run", help="run a flow by name, skipping completed steps")
    sp.add_argument("flow", help="flow name, e.g. align_only")
    sp.add_argument("rest", nargs="*")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("offers", help="vast offers matching the rental filters")
    sp.add_argument("--gpu", default="RTX_4090")
    sp.add_argument("--disk", type=int, default=100)
    sp.add_argument("--min-reliability", type=float, default=0.98)
    sp.add_argument("--min-inet-down", type=float, default=300.0)
    sp.add_argument("--max-dph", type=float, default=1.0)
    sp.add_argument("--limit", type=int, default=8)
    sp.set_defaults(fn=cmd_offers)

    sp = sub.add_parser("version", help="when the deployed code was last synced")
    sp.set_defaults(fn=cmd_version)

    sp = sub.add_parser("image", help="resolve an image tag to its registry digest")
    sp.add_argument("image")
    sp.set_defaults(fn=cmd_image)

    sp = sub.add_parser("boxes", help="live vast instances and their leases")
    sp.set_defaults(fn=cmd_boxes)

    sp = sub.add_parser("destroy", help="destroy an instance and drop its lease")
    sp.add_argument("instance_id", type=int)
    sp.set_defaults(fn=cmd_destroy)

    sp = sub.add_parser("reap", help="destroy instances past their lease expiry")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_reap)

    sp = sub.add_parser(
        "abort", help="kill a run's flow process, destroy its boxes, drop its leases"
    )
    sp.set_defaults(fn=cmd_abort)

    a = p.parse_args()
    a.cfg = Config.load()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

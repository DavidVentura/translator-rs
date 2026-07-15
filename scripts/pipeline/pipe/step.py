from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .job import Job, JobState
from .ledger import JobRef, Ledger
from .store import Store
from .target import IN_DIR, OUT_DIR, SCRIPTS_DIR, Target
from .types import Artifact, Kind, Name, RunId, Status, digest_file, digest_json


@dataclass(frozen=True)
class Ctx:
    script: Path
    out_dir: Path

    def inp(self, name: str) -> Path:
        return IN_DIR / name

    def out(self, rel: str) -> Path:
        return self.out_dir / rel


@dataclass(frozen=True)
class Output:
    rel: str
    kind: Kind


@dataclass(frozen=True)
class StepDef:
    name: str
    image: str
    target: Target
    script: str
    outputs: dict[str, Output]
    fn: Callable[[Ctx], list[str]]


def step(
    *,
    image: str,
    target: Target,
    script: str,
    outputs: dict[str, Output],
) -> Callable[[Callable[[Ctx], list[str]]], StepDef]:
    def deco(fn: Callable[[Ctx], list[str]]) -> StepDef:
        return StepDef(
            name=fn.__name__,
            image=image,
            target=target,
            script=script,
            outputs=outputs,
            fn=fn,
        )

    return deco


@dataclass
class Run:
    id: RunId
    store: Store
    ledger: Ledger
    scripts: Path

    def do(
        self,
        defn: StepDef,
        timeout: float,
        args: dict | None = None,
        **inputs: Artifact,
    ) -> dict[str, Artifact]:
        args = args or {}
        key = self._key(defn, inputs, args)

        cached = self.ledger.get(key)
        if cached is not None and cached.status is Status.DONE:
            return cached.outputs

        host = defn.target.host()
        job_dir = self.store.jobs_dir(self.id) / key
        out_dir = self.store.run_dir(self.id) / "out" / key
        in_paths = {name: art.path for name, art in inputs.items()}

        job = Job(host=host, ref=JobRef(host.name, job_dir), name=Name(defn.name))
        status = job.status()
        if status.state is JobState.EXITED and not status.ok:
            # An EXITED-ok job here means we died before collecting; that one must not relaunch.
            job.reset()
            status = job.status()
        if status.state is JobState.ABSENT:
            defn.target.materialize(in_paths, out_dir)
            payload = defn.fn(Ctx(script=SCRIPTS_DIR / defn.script, out_dir=OUT_DIR))
            exec_sh = defn.target.exec_sh(
                image=defn.image,
                job_dir=job_dir,
                scripts=self.scripts,
                inputs=in_paths,
                out_dir=out_dir,
            )
            self.ledger.begin(key, defn.name, job_dir / "log", job.ref)
            job.launch(exec_sh, [str(p) for p in payload], args)

        status = job.wait(timeout=timeout)
        if not status.ok:
            self.ledger.finish(key, Status.FAILED, {})
            raise RuntimeError(
                f"step {defn.name} exited {status.exit_code}; log: {job_dir / 'log'}"
            )

        produced = self._collect(defn, out_dir)
        self.ledger.finish(key, Status.DONE, produced)
        return produced

    def _collect(self, defn: StepDef, out_dir: Path) -> dict[str, Artifact]:
        produced: dict[str, Artifact] = {}
        for name, spec in defn.outputs.items():
            path = out_dir / spec.rel
            if not path.is_file():
                raise RuntimeError(
                    f"step {defn.name} exited 0 but produced no {spec.rel} at {path}"
                )
            produced[name] = self.store.describe(Name(name), path, spec.kind)
        return produced

    def _key(self, defn: StepDef, inputs: dict[str, Artifact], args: dict) -> str:
        script_path = self.scripts / defn.script
        if not script_path.is_file():
            raise FileNotFoundError(f"step {defn.name} names a missing script: {script_path}")
        payload = {
            "step": defn.name,
            "image": defn.target.image_id(defn.image),
            "script": str(digest_file(script_path)),
            "inputs": {k: str(v.digest) for k, v in sorted(inputs.items())},
            "args": args,
        }
        return f"{defn.name}@{digest_json(payload).short}"

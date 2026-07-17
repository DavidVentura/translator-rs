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
        # Before open(): a memoized step must not rent a box to find out it has nothing to do.
        key = self._key(defn, inputs, args)

        cached = self.ledger.get(key)
        if cached is not None and cached.status is Status.DONE:
            return cached.outputs

        # An operator asserted (via `pipe adopt`) that the edit which changed this
        # key cannot change the output, so reuse the previous run's artifacts rather
        # than pay for the step again. Recorded with provenance, never silent.
        inherited = self.ledger.take_adoption(defn.name)
        if inherited is not None:
            self.ledger.begin(key, defn.name, inherited.log, inherited.job)
            self.ledger.finish(key, Status.DONE, inherited.outputs, adopted_from=inherited.key)
            return inherited.outputs

        local_out = self.store.run_dir(self.id) / "out" / key
        in_paths = {name: art.path for name, art in inputs.items()}

        session = defn.target.open(
            run=str(self.id), step=defn.name, key=key, image=defn.image, root=self.store.root
        )
        ok = False
        try:
            job = Job(
                host=session.host,
                ref=JobRef(session.host.name, session.job_dir(key), session.ssh_info()),
                name=Name(defn.name),
            )
            status = job.status()
            if status.state is JobState.EXITED and not status.ok:
                # An EXITED-ok job here means we died before collecting; that must not relaunch.
                job.reset()
                status = job.status()
            if status.state is JobState.ABSENT:
                session.prepare(key, in_paths, self.scripts)
                payload = defn.fn(Ctx(script=SCRIPTS_DIR / defn.script, out_dir=OUT_DIR))
                self.ledger.begin(key, defn.name, local_out / "job.log", job.ref)
                job.launch(session.exec_sh(defn.image, key), [str(p) for p in payload], args)

            status = job.wait(timeout=timeout)
            session.collect(key, local_out)
            if not status.ok:
                self.ledger.finish(key, Status.FAILED, {})
                raise RuntimeError(
                    f"step {defn.name} exited {status.exit_code}; log: {local_out / 'job.log'}"
                )
            produced = self._collect(defn, local_out)
            ok = True
        finally:
            session.close(ok)

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

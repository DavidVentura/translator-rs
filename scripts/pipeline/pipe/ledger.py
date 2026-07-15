from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .types import Artifact, Name, Status


@dataclass(frozen=True)
class JobRef:
    host: str
    dir: Path

    def to_json(self) -> dict:
        return {"host": self.host, "dir": str(self.dir)}

    @staticmethod
    def from_json(d: dict) -> JobRef:
        return JobRef(host=d["host"], dir=Path(d["dir"]))


@dataclass(frozen=True)
class StepRecord:
    key: str
    step: str
    status: Status
    started: float
    finished: float | None
    log: Path
    job: JobRef | None
    outputs: dict[str, Artifact]

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "step": self.step,
            "status": self.status.value,
            "started": self.started,
            "finished": self.finished,
            "log": str(self.log),
            "job": self.job.to_json() if self.job else None,
            "outputs": {k: v.to_json() for k, v in self.outputs.items()},
        }

    @staticmethod
    def from_json(d: dict) -> StepRecord:
        return StepRecord(
            key=d["key"],
            step=d["step"],
            status=Status(d["status"]),
            started=d["started"],
            finished=d["finished"],
            log=Path(d["log"]),
            job=JobRef.from_json(d["job"]) if d["job"] else None,
            outputs={k: Artifact.from_json(v) for k, v in d["outputs"].items()},
        )


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.steps: dict[str, StepRecord] = {}
        self.artifacts: dict[str, Artifact] = {}
        if path.is_file():
            raw = json.loads(path.read_text())
            self.steps = {k: StepRecord.from_json(v) for k, v in raw["steps"].items()}
            self.artifacts = {k: Artifact.from_json(v) for k, v in raw["artifacts"].items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "steps": {k: v.to_json() for k, v in self.steps.items()},
            "artifacts": {k: v.to_json() for k, v in self.artifacts.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    def get(self, key: str) -> StepRecord | None:
        return self.steps.get(key)

    def begin(self, key: str, step: str, log: Path, job: JobRef | None) -> StepRecord:
        rec = StepRecord(
            key=key,
            step=step,
            status=Status.RUNNING,
            started=time.time(),
            finished=None,
            log=log,
            job=job,
            outputs={},
        )
        self.steps[key] = rec
        self.save()
        return rec

    def finish(self, key: str, status: Status, outputs: dict[str, Artifact]) -> StepRecord:
        rec = replace(
            self.steps[key],
            status=status,
            finished=time.time(),
            outputs=outputs,
        )
        self.steps[key] = rec
        for name, art in outputs.items():
            self.artifacts[name] = art
        self.save()
        return rec

    def register(self, artifact: Artifact) -> None:
        self.artifacts[str(artifact.name)] = artifact
        self.save()

    def artifact(self, name: Name) -> Artifact:
        got = self.artifacts.get(str(name))
        if got is None:
            known = ", ".join(sorted(self.artifacts)) or "none"
            raise KeyError(f"no artifact {name!s} in this run (have: {known})")
        return got

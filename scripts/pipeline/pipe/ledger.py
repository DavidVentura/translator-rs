from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .types import Artifact, Name, Status


@dataclass(frozen=True)
class SshInfo:
    host: str
    port: int
    user: str

    def to_json(self) -> dict:
        return {"host": self.host, "port": self.port, "user": self.user}

    @staticmethod
    def from_json(d: dict) -> SshInfo:
        return SshInfo(host=d["host"], port=d["port"], user=d["user"])


@dataclass(frozen=True)
class JobRef:
    host: str
    dir: Path
    # Lets `pipe log` reach a still-running job on a rented box with no flow process alive.
    ssh: SshInfo | None = None

    def to_json(self) -> dict:
        return {
            "host": self.host,
            "dir": str(self.dir),
            "ssh": self.ssh.to_json() if self.ssh else None,
        }

    @staticmethod
    def from_json(d: dict) -> JobRef:
        raw = d.get("ssh")
        return JobRef(
            host=d["host"], dir=Path(d["dir"]), ssh=SshInfo.from_json(raw) if raw else None
        )


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
        # Flows run parallel shards via threads; every mutation rewrites the whole
        # JSON, so mutations serialize behind one lock.
        self._lock = threading.RLock()
        self.steps: dict[str, StepRecord] = {}
        self.artifacts: dict[str, Artifact] = {}
        if path.is_file():
            raw = json.loads(path.read_text())
            self.steps = {k: StepRecord.from_json(v) for k, v in raw["steps"].items()}
            self.artifacts = {k: Artifact.from_json(v) for k, v in raw["artifacts"].items()}

    def save(self) -> None:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self.steps[key] = rec
            for name, art in outputs.items():
                self.artifacts[name] = art
            self.save()
        return rec

    def register(self, artifact: Artifact) -> None:
        with self._lock:
            self.artifacts[str(artifact.name)] = artifact
            self.save()

    def artifact(self, name: Name) -> Artifact:
        got = self.artifacts.get(str(name))
        if got is None:
            known = ", ".join(sorted(self.artifacts)) or "none"
            raise KeyError(f"no artifact {name!s} in this run (have: {known})")
        return got

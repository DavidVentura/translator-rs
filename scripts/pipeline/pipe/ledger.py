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
    # Set when this record's outputs were not produced under this key but adopted
    # from an earlier one, because an operator asserted the script edit was a no-op.
    # Kept in the ledger so an adoption is always visible when reading a run back.
    adopted_from: str | None = None

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
            "adopted_from": self.adopted_from,
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
            adopted_from=d.get("adopted_from"),
        )


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        # Flows run parallel shards via threads; every mutation rewrites the whole
        # JSON, so mutations serialize behind one lock.
        self._lock = threading.RLock()
        self.steps: dict[str, StepRecord] = {}
        self.artifacts: dict[str, Artifact] = {}
        # step name -> key whose outputs the next run of that step should adopt.
        # Consumed once, by the first cache miss for that step.
        self.adoptions: dict[str, str] = {}
        if path.is_file():
            raw = json.loads(path.read_text())
            self.steps = {k: StepRecord.from_json(v) for k, v in raw["steps"].items()}
            self.artifacts = {k: Artifact.from_json(v) for k, v in raw["artifacts"].items()}
            self.adoptions = raw.get("adoptions", {})

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "steps": {k: v.to_json() for k, v in self.steps.items()},
                "artifacts": {k: v.to_json() for k, v in self.artifacts.items()},
                "adoptions": self.adoptions,
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

    def finish(
        self, key: str, status: Status, outputs: dict[str, Artifact],
        adopted_from: str | None = None,
    ) -> StepRecord:
        rec = replace(
            self.steps[key],
            status=status,
            finished=time.time(),
            outputs=outputs,
            adopted_from=adopted_from,
        )
        with self._lock:
            self.steps[key] = rec
            for name, art in outputs.items():
                self.artifacts[name] = art
            self.save()
        return rec

    def adopt(self, step: str, old_key: str) -> None:
        """Declare that the next fresh key for `step` may reuse old_key's outputs.

        This exists because a step's key covers its script content, so an edit that
        cannot change the output — a retry, a timeout bump, added logging — still
        forces a re-run. Downstream is already safe without this: keys chain on
        input CONTENT digests, so a re-run producing identical bytes cascades
        nothing. Adoption only buys back the edited step's own runtime, which
        matters when that step is a paid decode or a many-hour train.

        The assertion that the edit is a no-op is the operator's, not the tool's,
        so it is recorded on the adopting record rather than applied silently.
        """
        with self._lock:
            rec = self.steps.get(old_key)
            if rec is None:
                raise KeyError(f"no step {old_key} in this run")
            if rec.status is not Status.DONE:
                raise ValueError(f"{old_key} is {rec.status.value}, not done; nothing to adopt")
            if rec.step != step:
                raise ValueError(f"{old_key} is step {rec.step!r}, not {step!r}")
            self.adoptions[step] = old_key
            self.save()

    def take_adoption(self, step: str) -> StepRecord | None:
        with self._lock:
            key = self.adoptions.pop(step, None)
            if key is None:
                return None
            self.save()
            return self.steps.get(key)

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

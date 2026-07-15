from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .host import Host
from .ledger import JobRef
from .types import Name

# exit_code is the only completion signal: a log marker gets scrolled past, and the
# tail-for-"Training finished" approach has already lost us a run.
WRAPPER = """#!/bin/bash
set -u
JOB=$1
cd "$JOB"
[ -f exit_code ] && exit 0
if [ -f pid ] && kill -0 "$(cat pid)" 2>/dev/null; then exit 0; fi
echo $$ > pid
: > log
bash exec.sh >> log 2>&1
echo $? > exit_code
"""


class JobState(Enum):
    ABSENT = "absent"
    RUNNING = "running"
    EXITED = "exited"


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    exit_code: int | None

    @property
    def ok(self) -> bool:
        return self.state is JobState.EXITED and self.exit_code == 0


@dataclass(frozen=True)
class Job:
    host: Host
    ref: JobRef
    name: Name

    @property
    def dir(self) -> Path:
        return self.ref.dir

    def status(self) -> JobStatus:
        code = self.host.read_file(self.dir / "exit_code")
        if code is not None:
            return JobStatus(JobState.EXITED, int(code.strip()))
        pid = self.host.read_file(self.dir / "pid")
        if pid is None:
            return JobStatus(JobState.ABSENT, None)
        if self.host.pid_alive(int(pid.strip())):
            return JobStatus(JobState.RUNNING, None)
        return JobStatus(JobState.ABSENT, None)

    def log(self) -> str:
        return self.host.read_file(self.dir / "log") or ""

    def reset(self) -> None:
        for leftover in ("exit_code", "pid", "log"):
            self.host.remove(self.dir / leftover)

    def launch(self, exec_sh: str, payload: list[str], args: dict) -> None:
        self.host.mkdir(self.dir)
        self.host.write_file(self.dir / "args.json", json.dumps(args, indent=2))
        self.host.write_file(self.dir / "cmd.sh", f"set -euo pipefail\n{shlex.join(payload)}\n")
        self.host.write_file(self.dir / "exec.sh", exec_sh)
        self.host.write_file(self.dir / "wrapper.sh", WRAPPER, executable=True)
        self.host.spawn(["setsid", "bash", str(self.dir / "wrapper.sh"), str(self.dir)])

    def wait(self, timeout: float, poll: float = 5.0) -> JobStatus:
        deadline = time.monotonic() + timeout
        while True:
            st = self.status()
            if st.state is JobState.EXITED:
                return st
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {self.name} still {st.state.value} after {timeout}s")
            time.sleep(poll)

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .host import Host, LocalHost

IN_DIR = Path("/work/in")
OUT_DIR = Path("/work/out")
JOB_DIR = Path("/work/job")
SCRIPTS_DIR = Path("/scripts")


class Target(Protocol):
    def host(self) -> Host: ...

    def image_id(self, image: str) -> str: ...

    def materialize(self, inputs: dict[str, Path], out_dir: Path) -> None: ...

    def exec_sh(
        self,
        image: str,
        job_dir: Path,
        scripts: Path,
        inputs: dict[str, Path],
        out_dir: Path,
    ) -> str: ...


@dataclass(frozen=True)
class Bigserver:
    cpus: int

    def host(self) -> Host:
        return LocalHost()

    def image_id(self, image: str) -> str:
        return LocalHost().capture(
            ["docker", "image", "inspect", "--format={{.Id}}", image]
        ).strip()

    def materialize(self, inputs: dict[str, Path], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, path in inputs.items():
            if not path.is_file():
                raise FileNotFoundError(f"input {name} missing at {path}")

    def exec_sh(
        self,
        image: str,
        job_dir: Path,
        scripts: Path,
        inputs: dict[str, Path],
        out_dir: Path,
    ) -> str:
        mounts: list[str] = []
        for name, path in sorted(inputs.items()):
            mounts += ["-v", f"{path}:{IN_DIR / name}:ro"]
        mounts += ["-v", f"{out_dir}:{OUT_DIR}"]
        mounts += ["-v", f"{job_dir}:{JOB_DIR}:ro"]
        mounts += ["-v", f"{scripts}:{SCRIPTS_DIR}:ro"]
        argv = [
            "docker", "run", "--rm",
            "--cpus", str(self.cpus),
            "--user", f"{_uid()}:{_gid()}",
            *mounts,
            "-w", str(OUT_DIR),
            image,
            "bash", str(JOB_DIR / "cmd.sh"),
        ]
        return f"set -u\n{shlex.join(argv)}\n"


def _uid() -> int:
    import os

    return os.getuid()


def _gid() -> int:
    import os

    return os.getgid()

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("~/.config/pipe/config.toml").expanduser()


@dataclass(frozen=True)
class Config:
    root: Path
    scripts: Path
    vast_key: Path

    @staticmethod
    def load(path: Path = DEFAULT_PATH) -> Config:
        if not path.is_file():
            raise FileNotFoundError(f"no config at {path} -- run sync.sh from the repo")
        raw = tomllib.loads(path.read_text())
        missing = {"root", "scripts", "vast_key"} - raw.keys()
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return Config(
            root=Path(raw["root"]).expanduser(),
            scripts=Path(raw["scripts"]).expanduser(),
            vast_key=Path(raw["vast_key"]).expanduser(),
        )

    def synced_at(self) -> str:
        stamp = self.root / "code" / "pipeline" / "SYNCED"
        if not stamp.is_file():
            return "unknown (no SYNCED stamp)"
        return stamp.read_text().strip()

"""Loads config.yaml (falling back to config.example.yaml) and .env secrets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Config(dict):
    """dict with attribute-free helpers; nested sections are plain dicts."""

    def path(self, section: str, key: str) -> Path:
        p = Path(self[section][key])
        return p if p.is_absolute() else (ROOT / p)

    @property
    def work_dir(self) -> Path:
        d = self.path("paths", "work_dir")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def sub(self, name: str) -> Path:
        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path | None = None) -> Config:
    _load_dotenv(ROOT / ".env")
    if path is None and os.environ.get("MOTO_SHORTS_CONFIG"):
        path = Path(os.environ["MOTO_SHORTS_CONFIG"])
    if path is None:
        path = ROOT / "config.yaml"
        if not path.exists():
            path = ROOT / "config.example.yaml"
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return Config(data)

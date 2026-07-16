"""Lightweight configuration for the MRC pipeline.

The config is just the packaged ``default.yaml`` deep-merged with an optional
user YAML, wrapped in a small object that supports both attribute access
(``cfg.recession.min_segment_length``) and dict access (``cfg["recession"]``).
No packaging, no per-field dataclasses — one file of thresholds, one loader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config:
    """Recursive namespace over a nested config dict.

    Supports ``cfg.section.key``, ``cfg["section"]``, ``.get(key, default)``,
    and ``.to_dict()`` for logging/serialization.
    """

    def __init__(self, data: dict):
        self._data = data
        for key, val in data.items():
            setattr(self, key, Config(val) if isinstance(val, dict) else val)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        out: dict = {}
        for key, val in self._data.items():
            out[key] = val.to_dict() if isinstance(val, Config) else val
        return out

    def dump_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: Optional[Path] = None, *, overrides: Optional[dict] = None) -> Config:
    """Load config, deep-merging user YAML (and optional overrides) over defaults."""
    with DEFAULT_CONFIG_PATH.open() as fh:
        merged = yaml.safe_load(fh) or {}
    if path is not None:
        with Path(path).open() as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    # Store the merged dict on each Config node so to_dict round-trips.
    return Config(merged)

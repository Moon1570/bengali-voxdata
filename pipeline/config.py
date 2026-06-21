"""Load and validate ``config/config.yaml`` into a typed ``Config`` object.

Stages receive a ``Config`` and read their section via attribute access, e.g.
``cfg.speakers.min_speakers`` or ``cfg.paths.manifest``. Unknown keys are
preserved so new config sections work without touching this loader.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import yaml

# Top-level sections that must be present for any run.
REQUIRED_SECTIONS = ("paths", "speakers")
# Required leaf keys within a section, checked after loading.
REQUIRED_KEYS = {
    "paths": ("data", "manifest"),
}


@dataclass
class Config:
    """Parsed configuration. Sections are accessible as attributes (namespaces)
    and the original mapping is kept on ``raw`` for forward compatibility."""

    raw: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails (i.e. for sections).
        raw = self.__dict__.get("raw", {})
        if name in raw:
            return _to_namespace(raw[name])
        raise AttributeError(f"no config section or key {name!r}")

    def get(self, section: str, default: Any = None) -> Any:
        """Return a section as a namespace, or ``default`` if absent."""
        if section in self.raw:
            return _to_namespace(self.raw[section])
        return default


def _to_namespace(value: Any) -> Any:
    """Recursively convert mappings to attribute-accessible namespaces."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path: str) -> Config:
    """Load ``path`` as YAML, validate required sections/keys, return a Config.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a required section or key is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"config missing required section(s): {', '.join(missing)}")
    for section, keys in REQUIRED_KEYS.items():
        absent = [k for k in keys if k not in (raw.get(section) or {})]
        if absent:
            raise ValueError(f"config section {section!r} missing key(s): {', '.join(absent)}")

    return Config(raw=raw, config_path=path)

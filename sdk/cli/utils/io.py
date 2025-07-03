from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Return YAML contents of ``path`` as a dict."""
    return yaml.safe_load(path.read_text())


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``path``."""
    path.write_text(json.dumps(data, indent=2))


def ensure_parent(path: Path) -> None:
    """Create parent directory of ``path`` if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["load_yaml", "write_json", "ensure_parent"]

"""Utility to load agent definitions from a catalog directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    """Return all agent definitions found in *path*."""
    agents: List[Dict[str, Any]] = []
    for file in path.glob("*.yaml"):
        with open(file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        data["_file"] = str(file)
        agents.append(data)
    return agents


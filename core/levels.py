from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

LEVEL_DIR = Path("levels")


@dataclass
class AgentLevel:
    """Definition of a career level for agents."""

    id: str
    title: str
    trust_required: float
    skills_required: List[str]
    unlocks: Dict[str, Any]


def load_level(level_id: str) -> AgentLevel | None:
    """Load ``AgentLevel`` from ``levels/{id}.json``."""
    LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    path = LEVEL_DIR / f"{level_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return AgentLevel(**data)


def save_level(level: AgentLevel) -> None:
    """Persist ``level`` in the registry."""
    LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    path = LEVEL_DIR / f"{level.id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(level), fh, indent=2)


def load_levels() -> List[AgentLevel]:
    """Return all defined levels ordered by ``trust_required``."""
    LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    levels: List[AgentLevel] = []
    for file in LEVEL_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        levels.append(AgentLevel(**data))
    levels.sort(key=lambda level: level.trust_required)
    return levels

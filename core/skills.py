from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

SKILL_REGISTRY = Path("skills")


@dataclass
class Skill:
    """Representation of a certifiable agent skill."""

    id: str
    title: str
    required_for_roles: List[str]
    expires_at: Optional[str] = None


def load_skill(skill_id: str) -> Skill | None:
    """Load a skill definition from ``skills/{id}.json``."""
    path = SKILL_REGISTRY / f"{skill_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Skill(**data)


def save_skill(skill: Skill) -> None:
    """Persist ``skill`` in the registry."""
    SKILL_REGISTRY.mkdir(parents=True, exist_ok=True)
    path = SKILL_REGISTRY / f"{skill.id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(skill), fh, indent=2)

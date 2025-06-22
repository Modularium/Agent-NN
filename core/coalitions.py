from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import json
import os
from uuid import uuid4

COALITION_DIR = Path(os.getenv("COALITION_DIR", "coalitions"))


@dataclass
class AgentCoalition:
    """Persistent representation of an agent coalition."""

    id: str
    goal: str
    leader: str
    members: List[str]
    strategy: str
    subtasks: List[Dict[str, Any]]

    @classmethod
    def load(cls, coalition_id: str) -> "AgentCoalition":
        COALITION_DIR.mkdir(parents=True, exist_ok=True)
        path = COALITION_DIR / f"{coalition_id}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(**data)
        return cls(
            id=coalition_id,
            goal="",
            leader="",
            members=[],
            strategy="plan-then-split",
            subtasks=[],
        )

    def save(self) -> None:
        COALITION_DIR.mkdir(parents=True, exist_ok=True)
        path = COALITION_DIR / f"{self.id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)


def create_coalition(
    goal: str,
    leader: str,
    members: List[str] | None = None,
    strategy: str = "plan-then-split",
) -> AgentCoalition:
    """Helper to create and persist a new coalition."""
    coalition_id = str(uuid4())
    coal = AgentCoalition(
        id=coalition_id,
        goal=goal,
        leader=leader,
        members=members or [],
        strategy=strategy,
        subtasks=[],
    )
    coal.save()
    return coal

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

TEAM_DIR = Path(os.getenv("TEAM_DIR", "teams"))


@dataclass
class AgentTeam:
    """Persistent representation of a team of agents."""

    id: str
    name: str
    members: List[str]
    shared_goal: Optional[str]
    skills_focus: List[str]
    coordinator: Optional[str]
    created_at: str

    @classmethod
    def load(cls, team_id: str) -> "AgentTeam":
        TEAM_DIR.mkdir(parents=True, exist_ok=True)
        path = TEAM_DIR / f"{team_id}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(**data)
        return cls(
            id=team_id,
            name="",
            members=[],
            shared_goal=None,
            skills_focus=[],
            coordinator=None,
            created_at=datetime.utcnow().isoformat(),
        )

    def save(self) -> None:
        TEAM_DIR.mkdir(parents=True, exist_ok=True)
        path = TEAM_DIR / f"{self.id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)


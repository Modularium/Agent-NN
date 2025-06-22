from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

MISSION_DIR = Path(os.getenv("MISSION_DIR", "missions"))


@dataclass
class AgentMission:
    """Definition of a multi-step agent mission."""

    id: str
    title: str
    description: str
    steps: List[Dict[str, Any]]
    rewards: Dict[str, Any]
    team_mode: Literal["solo", "team"] = "solo"
    mentor_required: bool = False
    track_id: Optional[str] = None

    @classmethod
    def load(cls, mission_id: str) -> "AgentMission | None":
        """Load a mission from ``missions/{id}.json``."""
        MISSION_DIR.mkdir(parents=True, exist_ok=True)
        path = MISSION_DIR / f"{mission_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    def save(self) -> None:
        """Persist the mission definition."""
        MISSION_DIR.mkdir(parents=True, exist_ok=True)
        path = MISSION_DIR / f"{self.id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

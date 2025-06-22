from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

PROFILE_DIR = Path(os.getenv("AGENT_PROFILE_DIR", "agent_profiles"))


@dataclass
class AgentIdentity:
    name: str
    role: str
    traits: Dict[str, Any]
    skills: List[str]
    memory_index: Optional[str]
    created_at: str
    estimated_cost_per_token: float = 0.0
    avg_response_time: float = 0.0
    load_factor: float = 0.0
    certified_skills: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, name: str) -> "AgentIdentity":
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILE_DIR / f"{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            defaults = asdict(
                cls(
                    name=name,
                    role="",
                    traits={},
                    skills=[],
                    memory_index=None,
                    created_at=datetime.now().isoformat(),
                    certified_skills=[],
                )
            )
            defaults.update(data)
            return cls(**defaults)
        return cls(
            name=name,
            role="",
            traits={},
            skills=[],
            memory_index=None,
            created_at=datetime.now().isoformat(),
            certified_skills=[],
        )

    def save(self) -> None:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILE_DIR / f"{self.name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    def update_metrics(
        self,
        cost_per_token: float | None = None,
        response_time: float | None = None,
        tasks_in_progress: int | None = None,
    ) -> None:
        """Update economic metrics and persist profile."""
        if cost_per_token is not None:
            self.estimated_cost_per_token = cost_per_token
        if response_time is not None:
            if self.avg_response_time:
                self.avg_response_time = (self.avg_response_time + response_time) / 2
            else:
                self.avg_response_time = response_time
        if tasks_in_progress is not None:
            self.load_factor = min(1.0, tasks_in_progress / 10)
        self.save()

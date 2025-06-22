from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .feedback_loop import FeedbackLoopEntry

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
    training_progress: Dict[str, str] = field(default_factory=dict)
    training_log: List[Dict[str, Any]] = field(default_factory=list)
    current_level: Optional[str] = None
    level_progress: Dict[str, Any] = field(default_factory=dict)
    team_id: Optional[str] = None
    team_role: Optional[str] = None
    shared_memory_scope: Optional[str] = None
    active_missions: List[str] = field(default_factory=list)
    mission_progress: Dict[str, Any] = field(default_factory=dict)
    reputation_score: float = 0.0
    feedback_log: List[Dict[str, Any]] = field(default_factory=list)
    trusted_by: List[str] = field(default_factory=list)
    endorsements: List[Dict[str, Any]] = field(default_factory=list)
    active_delegations: List[Dict[str, Any]] = field(default_factory=list)
    delegated_by: List[str] = field(default_factory=list)
    feedback_memory: List[FeedbackLoopEntry] = field(default_factory=list)
    adaptation_history: List[str] = field(default_factory=list)

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
                    training_progress={},
                    training_log=[],
                    current_level=None,
                    level_progress={},
                    team_id=None,
                    team_role=None,
                    shared_memory_scope=None,
                    active_missions=[],
                    mission_progress={},
                    reputation_score=0.0,
                    feedback_log=[],
                    trusted_by=[],
                    endorsements=[],
                    active_delegations=[],
                    delegated_by=[],
                    feedback_memory=[],
                    adaptation_history=[],
                )
            )
            defaults.update(data)
            defaults["feedback_memory"] = [
                FeedbackLoopEntry(**e) for e in defaults.get("feedback_memory", [])
            ]
            return cls(**defaults)
        return cls(
            name=name,
            role="",
            traits={},
            skills=[],
            memory_index=None,
            created_at=datetime.now().isoformat(),
            certified_skills=[],
            training_progress={},
            training_log=[],
            current_level=None,
            level_progress={},
            team_id=None,
            team_role=None,
            shared_memory_scope=None,
            active_missions=[],
            mission_progress={},
            reputation_score=0.0,
            feedback_log=[],
            trusted_by=[],
            endorsements=[],
            active_delegations=[],
            delegated_by=[],
            feedback_memory=[],
            adaptation_history=[],
        )

    def save(self) -> None:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILE_DIR / f"{self.name}.json"
        data = asdict(self)
        data["feedback_memory"] = [asdict(e) for e in self.feedback_memory]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

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

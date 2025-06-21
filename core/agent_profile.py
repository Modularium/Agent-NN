from __future__ import annotations

from dataclasses import dataclass, asdict
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

    @classmethod
    def load(cls, name: str) -> "AgentIdentity":
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILE_DIR / f"{name}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(**data)
        return cls(
            name=name,
            role="",
            traits={},
            skills=[],
            memory_index=None,
            created_at=datetime.now().isoformat(),
        )

    def save(self) -> None:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILE_DIR / f"{self.name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

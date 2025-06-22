from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from .agent_profile import AgentIdentity

RECOMMEND_DIR = Path(os.getenv("RECOMMEND_DIR", "recommendations"))


@dataclass
class AgentRecommendation:
    from_agent: str
    to_agent: str
    role: str
    confidence: float  # 0.0–1.0
    comment: Optional[str]
    created_at: str


def save_recommendation(rec: AgentRecommendation) -> None:
    """Append ``rec`` to ``recommendations/{to_agent}.jsonl``."""
    RECOMMEND_DIR.mkdir(parents=True, exist_ok=True)
    path = RECOMMEND_DIR / f"{rec.to_agent}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec)) + "\n")


def load_recommendations(agent_id: str) -> List[AgentRecommendation]:
    """Return all recommendations for ``agent_id``."""
    RECOMMEND_DIR.mkdir(parents=True, exist_ok=True)
    path = RECOMMEND_DIR / f"{agent_id}.jsonl"
    recs: List[AgentRecommendation] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                recs.append(AgentRecommendation(**data))
    return recs


def record_recommendation(rec: AgentRecommendation) -> None:
    """Persist ``rec`` and update the recipient profile."""
    save_recommendation(rec)
    profile = AgentIdentity.load(rec.to_agent)
    if rec.from_agent not in profile.trusted_by:
        profile.trusted_by.append(rec.from_agent)
    profile.endorsements.append(
        {
            "agent": rec.from_agent,
            "role": rec.role,
            "confidence": rec.confidence,
            "comment": rec.comment,
            "created_at": rec.created_at,
        }
    )
    profile.save()

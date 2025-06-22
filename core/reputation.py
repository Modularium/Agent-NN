from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from .agent_profile import AgentIdentity

RATING_DIR = Path(os.getenv("RATING_DIR", "ratings"))


@dataclass
class AgentRating:
    from_agent: str
    to_agent: str
    mission_id: Optional[str]
    rating: float  # 0.0–1.0
    feedback: Optional[str]
    context_tags: List[str]
    created_at: str


def save_rating(rating: AgentRating) -> None:
    """Append ``rating`` to ``ratings/{to_agent}.jsonl``."""
    RATING_DIR.mkdir(parents=True, exist_ok=True)
    path = RATING_DIR / f"{rating.to_agent}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rating)) + "\n")


def load_ratings(agent_id: str) -> List[AgentRating]:
    """Return all ratings for ``agent_id``."""
    RATING_DIR.mkdir(parents=True, exist_ok=True)
    path = RATING_DIR / f"{agent_id}.jsonl"
    ratings: List[AgentRating] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                ratings.append(AgentRating(**data))
    return ratings


def aggregate_score(agent_id: str) -> float:
    """Calculate mean rating for ``agent_id``."""
    ratings = load_ratings(agent_id)
    if not ratings:
        return 0.0
    return sum(r.rating for r in ratings) / len(ratings)


def update_reputation(agent_id: str) -> float:
    """Update profile reputation info and return the new score."""
    score = round(aggregate_score(agent_id), 3)
    profile = AgentIdentity.load(agent_id)
    profile.reputation_score = score
    profile.feedback_log = [asdict(r) for r in load_ratings(agent_id)[-10:]]
    profile.save()
    return score

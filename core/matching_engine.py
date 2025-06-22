from __future__ import annotations

import random
from typing import Any, Dict, List

from .agent_profile import AgentIdentity


def select_best_agent(
    subtask: Dict[str, Any],
    candidates: List[AgentIdentity],
    strategy: str = "balanced",
) -> str:
    """Return the name of the best matching agent."""
    skill = subtask.get("skill_required")
    filtered = [a for a in candidates if not skill or skill in a.skills]
    filtered = [a for a in filtered if a.load_factor < 0.8]
    if not filtered:
        filtered = candidates

    if not filtered:
        return ""

    if strategy == "random":
        return random.choice(filtered).name

    if strategy == "cost_first":
        filtered.sort(key=lambda a: (a.estimated_cost_per_token, a.avg_response_time))
    elif strategy == "skill_best":
        filtered.sort(
            key=lambda a: (0 if skill in a.skills else 1, a.avg_response_time)
        )
    elif strategy == "multi_criteria":
        filtered.sort(
            key=lambda a: (
                a.load_factor * 2
                + a.estimated_cost_per_token * 10
                + a.avg_response_time
            )
        )
    else:  # balanced
        filtered.sort(
            key=lambda a: (
                a.load_factor,
                a.estimated_cost_per_token,
                a.avg_response_time,
            )
        )

    return filtered[0].name

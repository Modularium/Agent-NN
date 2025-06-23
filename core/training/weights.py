from __future__ import annotations

from typing import Dict, Iterable


def accumulate_weights(entries: Iterable[tuple[str, float]]) -> Dict[str, float]:
    """Aggregate feedback scores per agent."""
    weights: Dict[str, float] = {}
    for agent_id, score in entries:
        weights[agent_id] = weights.get(agent_id, 0.0) + float(score)
    return weights

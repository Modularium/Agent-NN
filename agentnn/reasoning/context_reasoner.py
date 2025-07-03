"""Simple reasoning framework for agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningStep:
    """Information collected from a single agent run."""

    agent_id: str
    result: Any
    score: Optional[float] = None


class ContextReasoner:
    """Base class for collaborative reasoning."""

    def __init__(self) -> None:
        self.history: List[ReasoningStep] = []

    def add_step(self, agent_id: str, result: Any, score: float | None = None) -> None:
        """Store a reasoning step for later evaluation."""
        self.history.append(ReasoningStep(agent_id=agent_id, result=result, score=score))

    def decide(self) -> Any:
        """Return the aggregated decision for the collected steps."""
        raise NotImplementedError


class MajorityVoteReasoner(ContextReasoner):
    """Choose the result with the highest combined score."""

    def decide(self) -> Any:
        if not self.history:
            return None
        tally: Dict[Any, float] = {}
        for step in self.history:
            weight = step.score if step.score is not None else 1.0
            tally[step.result] = tally.get(step.result, 0.0) + weight
        return max(tally.items(), key=lambda item: item[1])[0]

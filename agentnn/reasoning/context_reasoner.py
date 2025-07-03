"""Simple reasoning framework for agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .tool_vote import BestToolSelector


@dataclass
class ReasoningStep:
    """Information collected from a single agent run."""

    agent_id: str
    result: Any
    score: Optional[float] = None
    role: str | None = None
    priority: int | None = None
    exclusive: bool = False


class ContextReasoner:
    """Base class for collaborative reasoning."""

    def __init__(self, allowed_roles: List[str] | None = None) -> None:
        self.history: List[ReasoningStep] = []
        self.allowed_roles = allowed_roles

    def add_step(
        self,
        agent_id: str,
        result: Any,
        score: float | None = None,
        *,
        role: str | None = None,
        priority: int | None = None,
        exclusive: bool = False,
    ) -> None:
        """Store a reasoning step for later evaluation."""
        self.history.append(
            ReasoningStep(
                agent_id=agent_id,
                result=result,
                score=score,
                role=role,
                priority=priority,
                exclusive=exclusive,
            )
        )

    def decide(self) -> Any:
        """Return the aggregated decision for the collected steps."""
        raise NotImplementedError


class MajorityVoteReasoner(ContextReasoner):
    """Choose the result with the highest combined score."""

    def decide(self) -> Any:
        steps = [
            s
            for s in self.history
            if not self.allowed_roles or (s.role in self.allowed_roles)
        ]
        if not steps:
            return None
        if any(s.exclusive for s in steps):
            steps = [s for s in steps if s.exclusive]
        steps.sort(key=lambda s: s.priority or 0, reverse=True)
        tally: Dict[Any, float] = {}
        for step in steps:
            weight = step.score if step.score is not None else 1.0
            tally[step.result] = tally.get(step.result, 0.0) + weight
        return max(tally.items(), key=lambda item: item[1])[0]


class ToolMajorityReasoner(ContextReasoner):
    """Evaluate tool results using majority vote on aggregated metrics."""

    def __init__(self, allowed_roles: List[str] | None = None) -> None:
        super().__init__(allowed_roles)
        self.tool_vote = BestToolSelector()

    def add_tool_result(
        self,
        tool: str,
        output: Any,
        *,
        confidence: float | None = None,
        relevance: float | None = None,
        consistency: float | None = None,
    ) -> None:
        """Register a tool result for later evaluation."""
        self.tool_vote.add_result(
            tool,
            output,
            confidence=confidence,
            relevance=relevance,
            consistency=consistency,
        )

    def decide(self) -> Any:
        result = self.tool_vote.decide()
        return result.output if result else None

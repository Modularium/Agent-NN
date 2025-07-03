"""Tool voting utilities for collaborative agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["ToolResult", "ToolResultVote", "BestToolSelector"]


@dataclass
class ToolResult:
    """Output of a single tool invocation."""

    tool: str
    output: Any
    confidence: float = 1.0
    relevance: float = 1.0
    consistency: float = 1.0


class ToolResultVote:
    """Collect and evaluate multiple tool results."""

    def __init__(self) -> None:
        self.results: List[ToolResult] = []

    def add_result(
        self,
        tool: str,
        output: Any,
        *,
        confidence: float | None = None,
        relevance: float | None = None,
        consistency: float | None = None,
    ) -> None:
        self.results.append(
            ToolResult(
                tool=tool,
                output=output,
                confidence=confidence or 1.0,
                relevance=relevance or 1.0,
                consistency=consistency or 1.0,
            )
        )

    def decide(self) -> Optional[ToolResult]:
        """Return the best tool result based on aggregated score."""
        if not self.results:
            return None
        scores: Dict[str, float] = {}
        for res in self.results:
            score = res.confidence + res.relevance + res.consistency
            scores[res.tool] = scores.get(res.tool, 0.0) + score
        best_tool = max(scores.items(), key=lambda x: x[1])[0]
        for res in reversed(self.results):
            if res.tool == best_tool:
                return res
        return None


class BestToolSelector(ToolResultVote):
    """Alias for ToolResultVote to match reasoning strategy naming."""

    pass

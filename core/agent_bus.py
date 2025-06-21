from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterator

_QUEUES: dict[str, deque] = defaultdict(deque)


def publish(agent_name: str, message: Dict[str, Any]) -> None:
    """Publish a message for an agent."""
    _QUEUES[agent_name].append(message)


def subscribe(agent_name: str) -> Iterator[Dict[str, Any]]:
    """Yield all pending messages for an agent."""
    q = _QUEUES[agent_name]
    while q:
        yield q.popleft()


def reset(agent_name: str) -> None:
    """Remove all pending messages for an agent."""
    _QUEUES[agent_name].clear()

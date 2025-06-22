from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class FeedbackLoopEntry:
    """Entry recording feedback or events relevant for adaptation."""

    agent_id: str
    event_type: str  # e.g. "task_failed", "criticism_received", "mission_success"
    data: Dict[str, Any]
    created_at: str


def _path(agent_id: str, base_dir: str) -> Path:
    return Path(base_dir) / f"{agent_id}.jsonl"


def record_feedback(entry: FeedbackLoopEntry, base_dir: str = "feedback_loops") -> None:
    """Append ``entry`` to the feedback log for ``agent_id``."""
    path = _path(entry.agent_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry)) + "\n")


def load_feedback(
    agent_id: str, base_dir: str = "feedback_loops"
) -> List[FeedbackLoopEntry]:
    """Load all feedback entries for ``agent_id``."""
    path = _path(agent_id, base_dir)
    if not path.exists():
        return []
    entries: List[FeedbackLoopEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            entries.append(FeedbackLoopEntry(**data))
    return entries

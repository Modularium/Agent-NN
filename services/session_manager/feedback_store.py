from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class FeedbackEntry:
    """Single feedback entry."""

    session_id: str
    user_id: str
    agent_id: str
    score: int
    comment: str | None = None
    timestamp: str = ""


class BaseFeedbackStore(ABC):
    """Interface for persisting feedback."""

    @abstractmethod
    def add_feedback(self, entry: FeedbackEntry) -> None:
        ...

    @abstractmethod
    def get_feedback(self, session_id: str) -> List[FeedbackEntry]:
        ...

    @abstractmethod
    def all_feedback(self) -> List[FeedbackEntry]:
        ...


class InMemoryFeedbackStore(BaseFeedbackStore):
    """Keep feedback in process memory."""

    def __init__(self) -> None:
        self._data: Dict[str, List[FeedbackEntry]] = {}

    def add_feedback(self, entry: FeedbackEntry) -> None:
        self._data.setdefault(entry.session_id, []).append(entry)

    def get_feedback(self, session_id: str) -> List[FeedbackEntry]:
        return list(self._data.get(session_id, []))

    def all_feedback(self) -> List[FeedbackEntry]:
        items: List[FeedbackEntry] = []
        for lst in self._data.values():
            items.extend(lst)
        return items


class FileFeedbackStore(BaseFeedbackStore):
    """Persist feedback as JSON files."""

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _file(self, sid: str) -> Path:
        return self.base_path / f"{sid}.json"

    def add_feedback(self, entry: FeedbackEntry) -> None:
        data = [asdict(e) for e in self.get_feedback(entry.session_id)]
        data.append(asdict(entry))
        with self._file(entry.session_id).open("w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def get_feedback(self, session_id: str) -> List[FeedbackEntry]:
        path = self._file(session_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return [FeedbackEntry(**d) for d in data]

    def all_feedback(self) -> List[FeedbackEntry]:
        items: List[FeedbackEntry] = []
        for file in self.base_path.glob("*.json"):
            items.extend(self.get_feedback(file.stem))
        return items

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseMemoryStore(ABC):
    """Abstract interface for persisting past interactions."""

    @abstractmethod
    def get_memory(self, session_id: str) -> List[Dict]:
        """Return stored memory entries for a session."""

    @abstractmethod
    def append_memory(self, session_id: str, entry: Dict) -> None:
        """Append a new memory entry."""


class InMemoryMemoryStore(BaseMemoryStore):
    """Keep memory in process memory."""

    def __init__(self) -> None:
        self._data: Dict[str, List[Dict]] = {}

    def get_memory(self, session_id: str) -> List[Dict]:
        return list(self._data.get(session_id, []))

    def append_memory(self, session_id: str, entry: Dict) -> None:
        self._data.setdefault(session_id, []).append(entry)


class FileMemoryStore(BaseMemoryStore):
    """Persist memory entries as JSON files."""

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self._cache: Dict[str, List[Dict]] = {}
        for fname in os.listdir(self.base_path):
            if fname.endswith(".json"):
                sid = fname[:-5]
                try:
                    with open(self._file(sid), encoding="utf-8") as f:
                        self._cache[sid] = json.load(f)
                except Exception:
                    self._cache[sid] = []

    def _file(self, sid: str) -> str:
        return os.path.join(self.base_path, f"{sid}.json")

    def get_memory(self, session_id: str) -> List[Dict]:
        if session_id in self._cache:
            return list(self._cache[session_id])
        path = self._file(session_id)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._cache[session_id] = data
        return list(data)

    def append_memory(self, session_id: str, entry: Dict) -> None:
        data = self.get_memory(session_id)
        data.append(entry)
        with open(self._file(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f)
        self._cache[session_id] = data


class NoOpMemoryStore(BaseMemoryStore):
    """Ignore memory operations."""

    def get_memory(self, session_id: str) -> List[Dict]:  # pragma: no cover - trivial
        return []

    def append_memory(self, session_id: str, entry: Dict) -> None:  # pragma: no cover - trivial
        pass


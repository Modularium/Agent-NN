from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseSessionStore(ABC):
    """Abstract session storage interface."""

    @abstractmethod
    def start_session(self) -> str:
        """Create and return a new session id."""

    @abstractmethod
    def get_context(self, session_id: str) -> List[Dict]:
        """Return stored ModelContext dicts for the session."""

    @abstractmethod
    def update_context(self, session_id: str, ctx: Dict) -> None:
        """Persist a ModelContext dict for the session."""


class InMemorySessionStore(BaseSessionStore):
    """Simple in-memory session store."""

    def __init__(self) -> None:
        self._data: Dict[str, List[Dict]] = {}

    def start_session(self) -> str:
        sid = str(uuid.uuid4())
        self._data[sid] = []
        return sid

    def get_context(self, session_id: str) -> List[Dict]:
        return list(self._data.get(session_id, []))

    def update_context(self, session_id: str, ctx: Dict) -> None:
        self._data.setdefault(session_id, []).append(ctx)


class FileSessionStore(BaseSessionStore):
    """Store sessions as JSON files in a directory."""

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _file(self, sid: str) -> str:
        return os.path.join(self.base_path, f"{sid}.json")

    def start_session(self) -> str:
        sid = str(uuid.uuid4())
        self._write(sid, [])
        return sid

    def _write(self, sid: str, data: List[Dict]) -> None:
        with open(self._file(sid), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def get_context(self, session_id: str) -> List[Dict]:
        path = self._file(session_id)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def update_context(self, session_id: str, ctx: Dict) -> None:
        data = self.get_context(session_id)
        data.append(ctx)
        self._write(session_id, data)


class NoOpSessionStore(BaseSessionStore):
    """Ignore all session operations."""

    def start_session(self) -> str:  # pragma: no cover - trivial
        return str(uuid.uuid4())

    def get_context(self, session_id: str) -> List[Dict]:  # pragma: no cover
        return []

    def update_context(self, session_id: str, ctx: Dict) -> None:  # pragma: no cover
        pass

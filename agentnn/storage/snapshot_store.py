"""Session snapshot utilities."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from . import context_store

_BASE_PATH = Path(os.getenv("SNAPSHOT_PATH", "data/snapshots"))


def save_snapshot(session_id: str, session_data: Dict[str, Any] | None = None) -> str:
    """Persist context and metadata for ``session_id`` and return snapshot id."""
    _BASE_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_id = str(uuid.uuid4())
    data = {
        "session_id": session_id,
        "context": context_store.load_context(session_id),
        "session": session_data or {},
    }
    with open(_BASE_PATH / f"{snapshot_id}.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return snapshot_id


def restore_snapshot(snapshot_id: str) -> str:
    """Restore snapshot and return new session id with loaded context."""
    path = _BASE_PATH / f"{snapshot_id}.json"
    if not path.exists():
        raise FileNotFoundError(snapshot_id)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    new_session = str(uuid.uuid4())
    for ctx in data.get("context", []):
        context_store.save_context(new_session, ctx)
    return new_session

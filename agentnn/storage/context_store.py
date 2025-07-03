"""Persistent context storage using SQLite or Redis."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Dict, List

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

__all__ = ["save_context", "load_context", "list_contexts"]

_BACKEND = os.getenv("CONTEXT_BACKEND", "sqlite").lower()
_DB_PATH = os.getenv("CONTEXT_DB_PATH", "data/context.db")
_REDIS_URL = os.getenv("CONTEXT_REDIS_URL", "redis://localhost:6379/0")

if _BACKEND == "redis" and redis is not None:
    _client = redis.Redis.from_url(_REDIS_URL)
    _backend = "redis"
else:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    _conn.execute(
        "CREATE TABLE IF NOT EXISTS contexts (session_id TEXT, ts REAL, data TEXT)"
    )
    _backend = "sqlite"


def save_context(session_id: str, ctx: Dict) -> None:
    """Persist a context for the given session."""
    payload = json.dumps(ctx)
    ts = time.time()
    if _backend == "redis":
        _client.rpush(f"ctx:{session_id}", payload)
    else:
        _conn.execute(
            "INSERT INTO contexts VALUES (?, ?, ?)", (session_id, ts, payload)
        )
        _conn.commit()


def load_context(session_id: str) -> List[Dict]:
    """Return stored contexts for a session."""
    if _backend == "redis":
        items = _client.lrange(f"ctx:{session_id}", 0, -1)
        return [json.loads(x) for x in items]
    rows = _conn.execute(
        "SELECT data FROM contexts WHERE session_id=? ORDER BY ts", (session_id,)
    ).fetchall()
    return [json.loads(r[0]) for r in rows]


def list_contexts() -> List[str]:
    """Return list of session ids that have stored contexts."""
    if _backend == "redis":
        keys = _client.keys("ctx:*")
        return [k.decode("utf-8")[4:] for k in keys]
    rows = _conn.execute("SELECT DISTINCT session_id FROM contexts").fetchall()
    return [r[0] for r in rows]

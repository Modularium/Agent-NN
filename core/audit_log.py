"""Simple append-only audit log utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json


@dataclass
class AuditEntry:
    """Single audit trail entry."""

    timestamp: str
    actor: str
    action: str
    context_id: str
    detail: Dict[str, Any]
    signature: str | None = None


class AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, log_dir: str = "audit") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        date = datetime.utcnow().date().isoformat()
        return self.log_dir / f"audit_{date}.log.jsonl"

    def write(self, entry: AuditEntry) -> str:
        """Write ``entry`` to today's log and return its id."""

        path = self._log_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry)) + "\n")
        return f"{path.stem}:{entry.timestamp}"

    def read_file(self, date: str) -> List[Dict[str, Any]]:
        """Return all entries for ``date`` (YYYY-MM-DD)."""

        path = self.log_dir / f"audit_{date}.log.jsonl"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def by_context(self, context_id: str) -> List[Dict[str, Any]]:
        """Return all entries matching ``context_id`` across logs."""

        entries: List[Dict[str, Any]] = []
        for file in self.log_dir.glob("audit_*.log.jsonl"):
            with file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("context_id") == context_id:
                        entries.append(data)
        return entries

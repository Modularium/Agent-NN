from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

HISTORY_DIR = Path.home() / ".agentnn" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def log_entry(command: str, params: Dict[str, Any]) -> None:
    """Append ``params`` for ``command`` to history."""
    entry = {"command": command, "params": params, "ts": datetime.utcnow().isoformat()}
    path = HISTORY_DIR / f"{command.replace(' ', '_')}.log"
    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


__all__ = ["log_entry"]

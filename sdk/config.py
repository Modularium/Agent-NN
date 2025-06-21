"""Configuration loader for the Developer SDK."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SDKSettings:
    """SDK configuration values."""

    host: str = "http://localhost:8000"
    api_token: Optional[str] = None

    @classmethod
    def load(cls) -> "SDKSettings":
        """Load settings from ``~/.agentnnrc`` and environment variables."""
        cfg = {}
        rc_path = Path.home() / ".agentnnrc"
        if rc_path.exists():
            try:
                cfg = json.loads(rc_path.read_text())
            except Exception:
                pass
        host = os.getenv("AGENTNN_HOST", cfg.get("host", cls.host))
        token = os.getenv("AGENTNN_API_TOKEN", cfg.get("api_token"))
        return cls(host=host, api_token=token)

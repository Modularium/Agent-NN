from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib
from ..config import SDKSettings as _SDKSettings


@dataclass
class CLIConfig:
    """User and project configuration values."""

    default_session_template: str = "examples/demo.yaml"
    output_format: str = "table"
    log_level: str = "INFO"
    templates_dir: str = "~/.agentnn/templates"

    @classmethod
    def load(cls) -> "CLIConfig":
        data: dict[str, str] = {}
        global_path = Path.home() / ".agentnn" / "config.toml"
        if global_path.exists():
            data.update(tomllib.loads(global_path.read_text()))
        project_path = Path.cwd() / "agentnn.toml"
        if project_path.exists():
            data.update(tomllib.loads(project_path.read_text()))
        if os.getenv("AGENTNN_LOG_LEVEL"):
            data["log_level"] = os.getenv("AGENTNN_LOG_LEVEL", "")

        return cls(**data)


SDKSettings = _SDKSettings

__all__ = ["CLIConfig", "SDKSettings"]

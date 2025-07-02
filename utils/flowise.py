"""Utility functions for Flowise compatibility."""

from __future__ import annotations

from typing import Any, Dict
from datetime import datetime


def agent_config_to_flowise(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal agent configuration to Flowise format.

    Parameters
    ----------
    config:
        Agent configuration dictionary.

    Returns
    -------
    dict
        Flowise compatible agent definition. Example::

            {
                "id": "demo",
                "name": "demo",
                "description": "demo domain",
                "type": "agent",
                "tools": [],
                "capabilities": [],
                "llm": {"model": "gpt-3.5"},
                "created_at": "2024-01-01T00:00:00",
                "version": "1.0.0"
            }
    """
    return {
        "id": config.get("name"),
        "name": config.get("name"),
        "description": config.get("domain", ""),
        "type": "agent",
        "tools": config.get("tools", []),
        "capabilities": config.get("capabilities", []),
        "llm": config.get("model_config", {}),
        "created_at": config.get(
            "created_at", datetime.now().isoformat()
        ),
        "version": config.get("version", "1.0.0"),
    }

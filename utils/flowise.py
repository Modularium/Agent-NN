"""Utility functions for Flowise compatibility."""

from __future__ import annotations

from typing import Any, Dict


def agent_config_to_flowise(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal agent configuration to Flowise format.

    Parameters
    ----------
    config:
        Agent configuration dictionary.

    Returns
    -------
    dict
        Flowise compatible agent definition.
    """
    return {
        "id": config.get("name"),
        "name": config.get("name"),
        "description": config.get("domain", ""),
        "type": "agent",
        "tools": config.get("tools", []),
        "capabilities": config.get("capabilities", []),
        "llm": config.get("model_config", {}),
        "created_at": config.get("created_at"),
        "version": config.get("version"),
    }

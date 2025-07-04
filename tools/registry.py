"""Lightweight in-memory registry for builtin tools."""

from __future__ import annotations

from typing import Dict, Type


class ToolRegistry:
    """Register and instantiate tool wrappers."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, tool_cls: Type) -> None:
        cls._registry[name] = tool_cls

    @classmethod
    def get(cls, name: str):
        tool_cls = cls._registry.get(name)
        return tool_cls() if tool_cls else None

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._registry.keys())

"""Service executing tool plugins."""

from __future__ import annotations

from typing import Any, Dict

from plugins import PluginManager


class PluginAgentService:
    """Load and execute registered tool plugins."""

    def __init__(self, plugin_dir: str = "plugins") -> None:
        self.manager = PluginManager(plugin_dir)

    def execute_tool(
        self, tool_name: str, input: Dict, context: Dict | None = None
    ) -> Dict:
        plugin = self.manager.get(tool_name)
        if not plugin:
            return {"error": "unknown tool"}
        ctx = context or {}
        return plugin.execute(input, ctx)

    def list_tools(self) -> list[str]:
        return self.manager.list_plugins()

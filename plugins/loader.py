"""Simple plugin loading utilities."""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Dict

import yaml


class ToolPlugin:
    """Base class for tool plugins."""

    def execute(self, input: dict, context: dict) -> dict:
        """Execute the tool with given input and context."""
        raise NotImplementedError


class PluginManager:
    """Discover and load tool plugins from a directory."""

    def __init__(self, plugin_dir: str = "plugins") -> None:
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, ToolPlugin] = {}
        self.load_plugins()

    def _load_module(self, path: str, name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:  # pragma: no cover - invalid spec
            raise ImportError(f"Cannot load module {name}")
        spec.loader.exec_module(module)
        return module

    def load_plugins(self) -> None:
        if not os.path.isdir(self.plugin_dir):
            return
        for entry in os.scandir(self.plugin_dir):
            if not entry.is_dir():
                continue
            manifest = os.path.join(entry.path, "manifest.yaml")
            plugin_py = os.path.join(entry.path, "plugin.py")
            if not os.path.exists(manifest) or not os.path.exists(plugin_py):
                continue
            with open(manifest, "r", encoding="utf-8") as fh:
                meta = yaml.safe_load(fh) or {}
            module = self._load_module(plugin_py, f"{entry.name}.plugin")
            plugin_cls = getattr(module, "Plugin", None)
            if plugin_cls is None:
                continue
            plugin = plugin_cls()
            name = meta.get("name", entry.name)
            self.plugins[name] = plugin

    def get(self, name: str) -> ToolPlugin | None:
        return self.plugins.get(name)

    def list_plugins(self) -> list[str]:
        return list(self.plugins.keys())

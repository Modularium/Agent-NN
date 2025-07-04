from __future__ import annotations

import json
import typer

from plugins.loader import PluginManager
from tools import ToolRegistry

tools_app = typer.Typer(name="tools", help="Tool registry")


@tools_app.command("list")
def tools_list() -> None:
    """List available plugins and builtin tools."""
    mgr = PluginManager()
    result = {
        "plugins": mgr.list_plugins(),
        "builtin": ToolRegistry.list_tools(),
    }
    typer.echo(json.dumps(result, indent=2))


@tools_app.command("inspect")
def tools_inspect(name: str) -> None:
    """Show details for ``name``."""
    mgr = PluginManager()
    plugin = mgr.get(name)
    if plugin:
        typer.echo(plugin.__class__.__name__)
        return
    tool = ToolRegistry.get(name)
    if not tool:
        typer.echo("not found")
        raise typer.Exit(1)
    typer.echo(tool.__class__.__name__)

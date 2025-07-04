from __future__ import annotations

import json
import typer

from plugins.loader import PluginManager

tools_app = typer.Typer(name="tools", help="Tool registry")


@tools_app.command("list")
def tools_list() -> None:
    """List available tool plugins."""
    mgr = PluginManager()
    typer.echo(json.dumps(mgr.list_plugins(), indent=2))


@tools_app.command("inspect")
def tools_inspect(name: str) -> None:
    """Show details for ``name``."""
    mgr = PluginManager()
    plugin = mgr.get(name)
    if not plugin:
        typer.echo("not found")
        raise typer.Exit(1)
    typer.echo(plugin.__class__.__name__)

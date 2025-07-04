from __future__ import annotations

import json

import typer

from plugins.loader import PluginManager

plugins_app = typer.Typer(name="plugins", help="Plugin utilities")


@plugins_app.command("list")
def plugins_list() -> None:
    """List installed plugins."""
    mgr = PluginManager()
    typer.echo(json.dumps(mgr.list_plugins(), indent=2))


@plugins_app.command("run")
def plugins_run(name: str, input: str = typer.Option("{}", "--input")) -> None:
    """Execute plugin ``name`` with JSON ``input``."""
    mgr = PluginManager()
    plugin = mgr.get(name)
    if not plugin:
        typer.echo("unknown plugin")
        raise typer.Exit(1)
    payload = json.loads(input)
    result = plugin.execute(payload, {})
    typer.echo(json.dumps(result, indent=2))


__all__ = ["plugins_app"]

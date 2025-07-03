"""Configuration commands."""

from __future__ import annotations

import json

import typer

from ..config import SDKSettings
from ..config import CLIConfig
from core.config import settings as core_settings

config_app = typer.Typer(name="config", help="Configuration commands")


@config_app.command("show")
def config_show() -> None:
    """Show effective configuration."""
    settings = SDKSettings.load()
    cli = CLIConfig.load()
    data = {"sdk": settings.__dict__, "cli": cli.__dict__}
    typer.echo(json.dumps(data, indent=2))


@config_app.command("check")
def config_check() -> None:
    """Validate and display core configuration."""
    typer.echo(json.dumps(core_settings.model_dump(), indent=2))


def register(app: typer.Typer) -> None:
    app.add_typer(config_app)


__all__ = ["register", "config_app"]

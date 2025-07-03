"""Helper utilities for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import httpx
import typer
import yaml


def handle_http_error(err: httpx.HTTPStatusError) -> None:
    """Show a friendly message for HTTP errors."""
    if err.response.status_code == 401:
        typer.secho(
            "\u26d4 Nicht autorisiert – überprüfe deinen API-Key",
            fg=typer.colors.RED,
        )
    else:
        typer.secho(f"HTTP Error: {err.response.status_code}", fg=typer.colors.RED)
    raise typer.Exit(1)


def print_success(msg: str) -> None:
    """Print a success message in green."""

    typer.secho(msg, fg=typer.colors.GREEN)


def print_error(msg: str) -> None:
    """Print an error message in red."""

    typer.secho(msg, fg=typer.colors.RED)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Return YAML contents of ``path`` as a dict."""

    return yaml.safe_load(path.read_text())


def confirm_action(prompt: str) -> bool:
    """Ask the user to confirm an action."""

    return typer.confirm(prompt)


__all__ = [
    "handle_http_error",
    "print_success",
    "print_error",
    "load_yaml",
    "confirm_action",
]

"""Utility helpers for CLI commands."""
from __future__ import annotations

import httpx
import os
import typer

from .formatting import print_success, print_error, print_output
from .io import load_yaml, write_json, ensure_parent


def handle_http_error(err: httpx.HTTPStatusError) -> None:
    """Show a friendly message for HTTP errors."""
    if err.response.status_code == 401:
        typer.secho(
            "\u26d4 Nicht autorisiert – überprüfe deinen API-Key",
            fg=typer.colors.RED,
        )
    else:
        typer.secho(
            f"HTTP Error: {err.response.status_code}", fg=typer.colors.RED
        )
    if os.getenv("AGENTNN_DEBUG") == "1":
        typer.echo(err.response.text)
        raise
    raise typer.Exit(1)


def confirm_action(prompt: str) -> bool:
    """Ask the user to confirm an action."""
    return typer.confirm(prompt)


__all__ = [
    "handle_http_error",
    "confirm_action",
    "print_success",
    "print_error",
    "print_output",
    "load_yaml",
    "write_json",
    "ensure_parent",
]

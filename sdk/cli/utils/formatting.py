from __future__ import annotations

import typer


def print_success(msg: str) -> None:
    """Print a success message in green."""
    typer.secho(msg, fg=typer.colors.GREEN)


def print_error(msg: str) -> None:
    """Print an error message in red."""
    typer.secho(msg, fg=typer.colors.RED)


__all__ = ["print_success", "print_error"]

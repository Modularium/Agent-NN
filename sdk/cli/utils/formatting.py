from __future__ import annotations

import json
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.table import Table


def print_success(msg: str) -> None:
    """Print a success message in green."""
    typer.secho(msg, fg=typer.colors.GREEN)


def print_error(msg: str) -> None:
    """Print an error message in red."""
    typer.secho(msg, fg=typer.colors.RED)


def print_output(data: List[Dict[str, Any]], output: str = "table") -> None:
    """Render ``data`` in the requested ``output`` format."""
    if output == "json":
        typer.echo(json.dumps(data, indent=2))
        return
    keys: List[str] = sorted({k for item in data for k in item.keys()})
    if output == "markdown":
        header = "| " + " | ".join(keys) + " |"
        sep = "| " + " | ".join("---" for _ in keys) + " |"
        rows = [
            "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |"
            for item in data
        ]
        typer.echo("\n".join([header, sep] + rows))
        return
    table = Table(*keys, show_header=True)
    for item in data:
        table.add_row(*(str(item.get(k, "")) for k in keys))
    Console().print(table)


__all__ = ["print_success", "print_error", "print_output"]

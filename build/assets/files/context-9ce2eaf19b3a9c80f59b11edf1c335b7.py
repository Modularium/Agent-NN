from __future__ import annotations

import json
from pathlib import Path

import typer

from agentnn.storage import context_store
from agentnn.context import context_map
from ..utils.formatting import print_success

context_app = typer.Typer(name="context", help="Context utilities")


@context_app.command("export")
def export(session_id: str, out: Path) -> None:
    """Export stored context for SESSION_ID to OUT."""
    data = context_store.load_context(session_id)
    out.write_text(json.dumps(data, indent=2))
    print_success(f"written to {out}")


@context_app.command("map")
def map_json(out: Path | None = None) -> None:
    """Generate a context map as JSON."""
    data = context_map.generate_map()
    if out:
        out.write_text(json.dumps(data, indent=2))
        print_success(f"written to {out}")
    else:
        typer.echo(json.dumps(data, indent=2))


@context_app.command("map-html")
def map_html(out: Path) -> None:
    """Generate a context map as interactive HTML."""
    context_map.export_html(out)
    print_success(f"written to {out}")


__all__ = ["context_app"]

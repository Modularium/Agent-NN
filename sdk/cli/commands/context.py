from __future__ import annotations

import json
from pathlib import Path

import typer

from agentnn.storage import context_store
from ..utils import print_success

context_app = typer.Typer(name="context", help="Context utilities")


@context_app.command("export")
def export(session_id: str, out: Path) -> None:
    """Export stored context for SESSION_ID to OUT."""
    data = context_store.load_context(session_id)
    out.write_text(json.dumps(data, indent=2))
    print_success(f"written to {out}")


__all__ = ["context_app"]

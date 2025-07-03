"""Task and queue management commands."""
from __future__ import annotations

import json

import typer

from ..client import AgentClient
from core.role_capabilities import apply_role_capabilities
from core.model_context import ModelContext, TaskContext

queue_app = typer.Typer(name="queue", help="Queue management")
task_app = typer.Typer(name="task", help="Task utilities")


@queue_app.command("status")
def queue_status() -> None:
    """Show current dispatch queue status."""
    client = AgentClient()
    resp = client._client.get("/queue/status")
    resp.raise_for_status()
    typer.echo(resp.text)


@task_app.command("limits")
def task_limits(context_file: str, role: str = typer.Option(..., "--role")) -> None:
    """Preview applied limits for CONTEXT_FILE and ROLE."""
    with open(context_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "task_type" in data:
        ctx = ModelContext(task_context=TaskContext(**data))
    else:
        ctx = ModelContext(**data)
    apply_role_capabilities(ctx, role)
    typer.echo(json.dumps(ctx.applied_limits, indent=2))


def register(app: typer.Typer) -> None:
    app.add_typer(queue_app)
    app.add_typer(task_app)

__all__ = ["register", "task_app", "queue_app"]

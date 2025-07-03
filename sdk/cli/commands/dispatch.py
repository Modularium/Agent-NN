import json

import typer

from core.model_context import ModelContext, TaskContext
from ...client import AgentClient


def register(app: typer.Typer) -> None:
    """Register dispatch command on ``app``."""

    @app.command("dispatch")
    def dispatch(task: str) -> None:
        """Send TASK to the dispatcher and output the result."""
        client = AgentClient()
        ctx = ModelContext(task_context=TaskContext(task_type="chat", description=task))
        result = client.dispatch_task(ctx)
        typer.echo(json.dumps(result))


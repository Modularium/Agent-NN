import json
import sys

import typer

from core.model_context import ModelContext, TaskContext
from ...client import AgentClient


def register(app: typer.Typer) -> None:
    """Register dispatch command on ``app``."""

    @app.command("dispatch")
    def dispatch(
        task: str = "",
        from_stdin: bool = typer.Option(
            False, "--from-stdin", help="read JSON task from stdin"
        ),
    ) -> None:
        """Send ``task`` to the dispatcher and output the result."""
        if from_stdin:
            data = json.loads(sys.stdin.read() or "{}")
            task = data.get("task", task)
        if not task:
            typer.echo("missing task")
            raise typer.Exit(1)
        client = AgentClient()
        ctx = ModelContext(task_context=TaskContext(task_type="chat", description=task))
        result = client.dispatch_task(ctx)
        typer.echo(json.dumps(result))

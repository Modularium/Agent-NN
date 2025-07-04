import json
import sys

import typer

from core.model_context import ModelContext, TaskContext
from services import dispatch_task


def register(app: typer.Typer) -> None:
    """Register dispatch command on ``app``."""

    @app.command("dispatch")
    def dispatch(
        task: str = "",
        tool: str = typer.Option(
            None,
            "--tool",
            "--model",
            "--reasoner",
            "--nn",
            help="Execute task using TOOL",
        ),
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
        ctx = ModelContext(task_context=TaskContext(task_type="chat", description=task))
        if tool:
            ctx.agent_selection = tool
        result = dispatch_task(ctx)
        typer.echo(json.dumps(result))

"""Command line tool for Agent-NN."""
from __future__ import annotations

import json
import typer

from ..client import AgentClient
from ..config import SDKSettings

app = typer.Typer()


@app.command()
def submit(task: str) -> None:
    """Submit a task to the dispatcher."""
    client = AgentClient()
    result = client.submit_task(task)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def sessions() -> None:
    """List active sessions."""
    client = AgentClient()
    result = client.list_sessions()
    typer.echo(json.dumps(result, indent=2))


@app.command(name="agent")
def agents() -> None:
    """List available agents."""
    client = AgentClient()
    result = client.list_agents()
    typer.echo(json.dumps(result, indent=2))


@app.command()
def config() -> None:
    """Show effective configuration."""
    settings = SDKSettings.load()
    typer.echo(json.dumps(settings.__dict__, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

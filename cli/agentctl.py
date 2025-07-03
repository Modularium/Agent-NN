from __future__ import annotations

"""Command line tool to deploy agents via the MCP API."""

from pathlib import Path
import json

import typer

from agentnn.deployment.agent_registry import AgentRegistry, load_agent_file

app = typer.Typer(help="Manage agent deployments")


@app.command()
def deploy(config: Path, endpoint: str = "http://localhost:8090") -> None:
    """Deploy the agent described in CONFIG."""
    data = load_agent_file(config)
    registry = AgentRegistry(endpoint)
    result = registry.deploy(data)
    typer.echo(json.dumps(result))


if __name__ == "__main__":  # pragma: no cover - manual use
    app()


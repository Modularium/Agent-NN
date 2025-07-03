import json
from pathlib import Path

import typer

from agentnn.deployment.agent_registry import AgentRegistry, load_agent_file


def register(app: typer.Typer) -> None:
    """Register deployment commands on ``app``."""

    @app.command("deploy")
    def deploy(config: Path, endpoint: str = "http://localhost:8090") -> None:
        """Deploy the agent described in CONFIG."""
        data = load_agent_file(config)
        registry = AgentRegistry(endpoint)
        result = registry.deploy(data)
        typer.echo(json.dumps(result))


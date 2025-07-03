from __future__ import annotations

from pathlib import Path
import typer
import yaml

from .session import start as session_start
from ..config import CLIConfig
from ..utils.io import ensure_parent


quickstart_app = typer.Typer(name="quickstart", help="Automated setup helpers")


@quickstart_app.command("agent")
def quickstart_agent(
    name: str = typer.Option(...),
    role: str = typer.Option("planner"),
    output: Path = typer.Option(Path("agent.yaml")),
) -> None:
    """Create a new agent config from template."""
    built_in = (
        Path(__file__).resolve().parent.parent / "templates" / "agent_template.yaml"
    )
    data = yaml.safe_load(built_in.read_text())
    data["id"] = name
    data["role"] = role
    ensure_parent(output)
    output.write_text(yaml.safe_dump(data))
    typer.echo(str(output))


@quickstart_app.command("session")
def quickstart_session(template: str | None = None) -> None:
    """Start a session using a template."""
    cfg = CLIConfig.load()
    path = Path(template or cfg.default_session_template)
    session_start(template=path)


__all__ = ["quickstart_app"]

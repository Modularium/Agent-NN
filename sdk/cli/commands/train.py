from __future__ import annotations

import typer
from datetime import datetime

from core.agent_profile import AgentIdentity
from core.audit_log import AuditEntry, AuditLog

train_app = typer.Typer(name="train", help="Training management")


@train_app.command("start")
def train_start(
    agent: str,
    path: str = typer.Option(..., "--path"),
    interactive: bool = typer.Option(False, "--interactive"),
) -> None:
    """Start a training path for ``agent``."""
    if interactive:
        agent = typer.prompt("Agent", default=agent)
        path = typer.prompt("Path", default=path)
    profile = AgentIdentity.load(agent)
    profile.training_progress[path] = "in_progress"
    profile.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="training_started",
            context_id=agent,
            detail={"path": path},
        )
    )
    typer.echo("started")

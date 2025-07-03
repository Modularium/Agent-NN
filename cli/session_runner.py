"""Interactive CLI for running Agent-NN sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
import yaml

from agentnn.reasoning.context_reasoner import MajorityVoteReasoner
from agentnn.session.session_manager import SessionManager
from agentnn.storage import snapshot_store

app = typer.Typer(help="Run and observe live agent sessions")
manager = SessionManager()


@app.command()
def start(template: Optional[Path] = None) -> None:
    """Start a new session from optional YAML template."""
    sid = manager.create_session()
    if template and template.exists():
        data = yaml.safe_load(template.read_text())
        for agent in data.get("agents", []):
            manager.add_agent(
                sid,
                agent["id"],
                role=agent.get("role"),
                priority=agent.get("priority", 1),
                exclusive=agent.get("exclusive", False),
            )
        for task in data.get("tasks", []):
            manager.run_task(sid, task)
    typer.echo(json.dumps({"session_id": sid}))


@app.command()
def watch(session_id: str) -> None:
    """Print message history for a session."""
    session = manager.get_session(session_id)
    typer.echo(json.dumps(session.get("message_history", []), indent=2))


@app.command()
def vote(session_id: str, roles: str = "") -> None:
    """Aggregate last step results using majority vote."""
    session = manager.get_session(session_id)
    reasoner = MajorityVoteReasoner(
        [r.strip() for r in roles.split(",") if r.strip()] or None
    )
    for entry in session.get("message_history", []):
        reasoner.add_step(
            entry["agent"],
            entry.get("result"),
            role=entry.get("role"),
        )
    typer.echo(json.dumps({"decision": reasoner.decide()}))


@app.command()
def refine(session_id: str, message: str) -> None:
    """Run another task with all agents."""
    manager.run_task(session_id, message)
    typer.echo("ok")


@app.command()
def snapshot(session_id: str) -> None:
    """Save snapshot of session state."""
    session = manager.get_session(session_id) or {}
    snap = snapshot_store.save_snapshot(session_id, session)
    typer.echo(json.dumps({"snapshot_id": snap}))


@app.command()
def restore(snapshot_id: str) -> None:
    """Restore a snapshot and return new session id."""
    sid = snapshot_store.restore_snapshot(snapshot_id)
    typer.echo(json.dumps({"session_id": sid}))


if __name__ == "__main__":
    app()

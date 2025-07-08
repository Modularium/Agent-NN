from __future__ import annotations

import json
from datetime import datetime
import typer

from ...client import AgentClient

feedback_app = typer.Typer(name="feedback", help="Feedback utilities")


@feedback_app.command("record")
def feedback_record(
    session: str = "",
    score: int = typer.Option(..., "--score"),
    comment: str = typer.Option("", "--comment"),
    agent: str = typer.Option(None, "--agent"),
    interactive: bool = typer.Option(False, "--interactive"),
) -> None:
    """Send a feedback entry."""
    if interactive:
        if not session:
            session = typer.prompt("Session id")
        if not score:
            score = int(typer.prompt("Score"))
        comment = typer.prompt("Comment", default=comment)
    payload = {
        "session_id": session,
        "user_id": "default",
        "agent_id": agent or "",
        "score": score,
        "comment": comment or None,
        "timestamp": datetime.utcnow().isoformat(),
    }
    client = AgentClient()
    result = client.post_feedback(session, payload)
    typer.echo(json.dumps(result))


@feedback_app.command("list")
def feedback_list(session: str) -> None:
    """List feedback for SESSION."""
    client = AgentClient()
    result = client.get_feedback(session)
    typer.echo(json.dumps(result, indent=2))

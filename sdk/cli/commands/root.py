"""Root level CLI commands."""
from __future__ import annotations

import json
import os
from datetime import datetime

import typer
import httpx

from .. import __version__
from ..client import AgentClient
from ..utils import handle_http_error
from core.audit_log import AuditEntry, AuditLog
from core.crypto import verify_signature
from core.model_context import ModelContext
from core.reputation import AgentRating, save_rating, update_reputation

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
    token: str = typer.Option(None, "--token", help="API token"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress info output"),
    debug: bool = typer.Option(False, "--debug", help="Show stack traces"),
) -> None:
    """Global options."""
    if version:
        typer.echo(__version__)
        ctx.exit()
    if token:
        os.environ["AGENTNN_API_TOKEN"] = token
    if debug:
        os.environ["AGENTNN_DEBUG"] = "1"
    level = "INFO"
    if verbose:
        level = "DEBUG"
    if quiet:
        level = "WARNING"
    os.environ.setdefault("AGENTNN_LOG_LEVEL", level)


@app.command()
def submit(
    task: str,
    value: float = typer.Option(None, "--value"),
    max_tokens: int = typer.Option(None, "--max-tokens"),
    priority: int = typer.Option(None, "--priority"),
    deadline: str = typer.Option(None, "--deadline"),
) -> None:
    """Submit a task to the dispatcher."""
    client = AgentClient()
    result = client.submit_task(
        task,
        value=value,
        max_tokens=max_tokens,
        priority=priority,
        deadline=deadline,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def ask(
    message: str,
    task_type: str = typer.Option("dev", "--task-type"),
    verbose_routing: bool = typer.Option(
        False, "--verbose-routing", help="Show routing decisions"
    ),
) -> None:
    """Send a quick task to the dispatcher."""
    client = AgentClient()
    if verbose_routing:
        try:
            resp = httpx.post(
                "http://localhost:8111/route",
                json={"task_type": task_type},
                timeout=5,
            )
            resp.raise_for_status()
            typer.echo(f"Routing to: {resp.json().get('target_worker')}")
        except Exception:
            typer.echo("Routing info unavailable")
    result = client.chat(message, task_type=task_type)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def sessions() -> None:
    """List active sessions."""
    client = AgentClient()
    try:
        result = client.list_sessions()
    except httpx.HTTPStatusError as err:
        handle_http_error(err)
        return
    typer.echo(json.dumps(result, indent=2))


@app.command("rate")
def rate_agent(
    from_agent: str,
    to_agent: str,
    score: float = typer.Option(..., "--score"),
    tags: str = typer.Option("", "--tags"),
    mission_id: str = typer.Option(None, "--mission-id"),
    feedback: str = typer.Option(None, "--feedback"),
) -> None:
    """Submit a peer rating."""
    if not 0.0 <= score <= 1.0:
        log = AuditLog()
        log.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor=from_agent,
                action="rating_declined",
                context_id=to_agent,
                detail={"score": score},
            )
        )
        typer.echo("invalid score")
        raise typer.Exit(code=1)
    rating = AgentRating(
        from_agent=from_agent,
        to_agent=to_agent,
        mission_id=mission_id,
        rating=score,
        feedback=feedback,
        context_tags=[t.strip() for t in tags.split(",") if t.strip()],
        created_at=datetime.utcnow().isoformat(),
    )
    save_rating(rating)
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor=from_agent,
            action="peer_rated",
            context_id=to_agent,
            detail={"score": score},
        )
    )
    new_score = update_reputation(to_agent)
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="system",
            action="reputation_updated",
            context_id=to_agent,
            detail={"score": new_score},
        )
    )
    typer.echo("recorded")


@app.command("verify")
def verify_ctx(context_json: str) -> None:
    """Verify the signature of a ModelContext JSON file."""
    with open(context_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ctx = ModelContext(**data)
    if ctx.signed_by and ctx.signature:
        payload = ctx.model_dump(exclude={"signature", "signed_by"})
        valid = verify_signature(ctx.signed_by, payload, ctx.signature)
    else:
        valid = False
    typer.echo(json.dumps({"valid": valid}))


@app.command("promote")
def task_promote(task_id: str) -> None:
    """Promote a queued task by id."""
    client = AgentClient()
    resp = client._client.post(f"/queue/promote/{task_id}")
    resp.raise_for_status()
    typer.echo(resp.text)


__all__ = ["app"]

"""Governance and contract related commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import typer

from core.governance import AgentContract
from core.trust_evaluator import calculate_trust, eligible_for_role
from core.audit_log import AuditEntry, AuditLog
from core.privacy_filter import redact_context
from core.model_context import ModelContext, TaskContext
from core.trust_network import (
    AgentRecommendation,
    record_recommendation,
    load_recommendations,
)
from core.trust_circle import is_trusted_for
from core.voting import ProposalVote, record_vote, load_votes

contract_app = typer.Typer(name="contract", help="Governance contracts")
trust_app = typer.Typer(name="trust", help="Trust management")
privacy_app = typer.Typer(name="privacy", help="Privacy tools")
audit_app = typer.Typer(name="audit", help="Audit utilities")
auth_app = typer.Typer(name="auth", help="Authorization utilities")
role_app = typer.Typer(name="role", help="Role utilities")
governance_app = typer.Typer(name="governance", help="High level governance utilities")


@contract_app.command("view")
def contract_view(agent: str) -> None:
    """Show contract for an agent."""
    contract = AgentContract.load(agent)
    typer.echo(json.dumps(asdict(contract), indent=2))


@contract_app.command("audit")
def contract_audit() -> None:
    """List agents violating trust requirements."""
    path = Path(os.getenv("CONTRACT_DIR", "contracts"))
    violations = []
    for file in path.glob("*.json"):
        c = AgentContract.load(file.stem)
        score = calculate_trust(c.agent, [])
        if score < c.trust_level_required:
            violations.append(c.agent)
    typer.echo(json.dumps({"violations": violations}, indent=2))


@contract_app.command("temp-role")
def contract_temp_role(agent: str, grant: str = typer.Option(..., "--grant")) -> None:
    """Grant a temporary role valid for one task."""
    contract = AgentContract.load(agent)
    contract.temp_roles = contract.temp_roles or []
    if grant not in contract.temp_roles:
        contract.temp_roles.append(grant)
        contract.save()
    typer.echo("granted")


@trust_app.command("score")
def trust_score(agent: str) -> None:
    """Calculate trust score for an agent."""
    score = calculate_trust(agent, [])
    typer.echo(json.dumps({"agent": agent, "score": score}, indent=2))


@trust_app.command("eligible")
def trust_eligible(agent: str, for_role: str = typer.Option(..., "--for")) -> None:
    """Check if AGENT may be elevated to FOR_ROLE."""
    allowed = eligible_for_role(agent, for_role)
    typer.echo(
        json.dumps({"agent": agent, "role": for_role, "eligible": allowed}, indent=2)
    )


@trust_app.command("endorse")
def trust_endorse(
    from_agent: str,
    to_agent: str,
    role: str = typer.Option(..., "--role"),
    confidence: float = typer.Option(1.0, "--confidence"),
    comment: str = typer.Option(None, "--comment"),
) -> None:
    """Record a recommendation from FROM_AGENT to TO_AGENT."""
    if not 0.0 <= confidence <= 1.0:
        typer.echo("invalid confidence")
        raise typer.Exit(code=1)
    rec = AgentRecommendation(
        from_agent=from_agent,
        to_agent=to_agent,
        role=role,
        confidence=confidence,
        comment=comment,
        created_at=datetime.utcnow().isoformat(),
    )
    record_recommendation(rec)
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor=from_agent,
            action="collective_endorsement_granted",
            context_id=to_agent,
            detail={"role": role, "confidence": confidence},
        )
    )
    typer.echo("recorded")


@trust_app.command("circle")
def trust_circle(name: str) -> None:
    """Show roles for which AGENT is trusted."""
    recs = load_recommendations(name)
    roles = {r.role for r in recs}
    result = {role: is_trusted_for(name, role) for role in roles}
    typer.echo(json.dumps(result, indent=2))


@privacy_app.command("preview")
def privacy_preview(agent: str, task_file: str) -> None:
    """Show context after privacy filtering for an agent."""
    with open(task_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ctx = ModelContext(task_context=TaskContext(**data))
    contract = AgentContract.load(agent)
    filtered = redact_context(ctx, contract.max_access_level)
    typer.echo(json.dumps(filtered.model_dump(), indent=2))


@audit_app.command("view")
def audit_view(context_id: str) -> None:
    """Show audit entries for a context."""
    log = AuditLog()
    typer.echo(json.dumps(log.by_context(context_id), indent=2))


@audit_app.command("log")
def audit_log(date: str) -> None:
    """Print all entries for a date."""
    log = AuditLog()
    typer.echo(json.dumps(log.read_file(date), indent=2))


@audit_app.command("violations")
def audit_violations() -> None:
    """Show logged access violations."""
    log = AuditLog()
    entries = []
    for file in log.log_dir.glob("audit_*.log.jsonl"):
        entries.extend(
            e
            for e in log.read_file(file.stem.replace("audit_", ""))
            if e.get("detail", {}).get("access_violated")
        )
    typer.echo(json.dumps(entries, indent=2))


@auth_app.command("check")
def auth_check(agent: str, role: str, action: str, resource: str) -> None:
    """Check if AGENT with ROLE may perform ACTION."""
    from core.access_control import is_authorized

    allowed = is_authorized(agent, role, action, resource)
    typer.echo(json.dumps({"authorized": allowed}))


@role_app.command("limits")
def role_limits(role: str) -> None:
    """Show resource limits for ROLE."""
    from core.role_capabilities import ROLE_CAPABILITIES

    typer.echo(json.dumps({role: ROLE_CAPABILITIES.get(role, {})}, indent=2))


@governance_app.command("vote")
def governance_vote(
    proposal: str,
    agent: str = typer.Option(..., "--agent"),
    decision: str = typer.Option(..., "--decision", help="yes|no"),
    comment: str = typer.Option(None, "--comment"),
) -> None:
    """Record a vote for ``proposal`` by ``agent``."""
    vote = ProposalVote(
        proposal_id=proposal,
        agent_id=agent,
        decision=decision,
        comment=comment,
        created_at=datetime.utcnow().isoformat(),
    )
    record_vote(vote)
    typer.echo("recorded")


@governance_app.command("log")
def governance_log(proposal: str) -> None:
    """Show all votes for ``proposal``."""
    votes = [asdict(v) for v in load_votes(proposal)]
    typer.echo(json.dumps(votes, indent=2))


@governance_app.command("audit")
def governance_audit() -> None:
    """Alias for :func:`contract_audit`."""
    contract_audit()


def register(app: typer.Typer) -> None:
    app.add_typer(governance_app)
    governance_app.add_typer(contract_app)
    governance_app.add_typer(trust_app)
    governance_app.add_typer(privacy_app)
    governance_app.add_typer(audit_app)
    governance_app.add_typer(auth_app)
    governance_app.add_typer(role_app)


__all__ = ["register"]

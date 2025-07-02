"""Command line tool for Agent-NN."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx
import mlflow
import typer


def _handle_http_error(err: httpx.HTTPStatusError) -> None:
    """Show a friendly message for HTTP errors."""
    if err.response.status_code == 401:
        typer.secho(
            "\u26d4 Nicht autorisiert – überprüfe deinen API-Key", fg=typer.colors.RED
        )
    else:
        typer.secho(f"HTTP Error: {err.response.status_code}", fg=typer.colors.RED)
    raise typer.Exit(1)


from core.access_control import is_authorized
from core.agent_evolution import evolve_profile
from core.agent_profile import PROFILE_DIR, AgentIdentity
from core.audit_log import AuditEntry, AuditLog
from core.crypto import generate_keypair, verify_signature
from core.governance import AgentContract
from core.level_evaluator import check_level_up
from core.model_context import ModelContext, TaskContext
from core.privacy import AccessLevel
from core.privacy_filter import redact_context
from core.reputation import AgentRating, aggregate_score, save_rating, update_reputation
from core.skills import load_skill
from core.trust_evaluator import auto_certify, calculate_trust, eligible_for_role

from .. import __version__
from ..client import AgentClient
from ..config import SDKSettings
from ..nn_models import ModelManager

model_app = typer.Typer(name="model", help="Model management commands")
config_app = typer.Typer(name="config", help="Configuration commands")

app = typer.Typer()
agent_app = typer.Typer(name="agent", help="Agent management")
agent_contract_app = typer.Typer(name="contract", help="Agent contract management")
team_app = typer.Typer(name="team", help="Team management")
session_app = typer.Typer(name="session", help="Session utilities")
queue_app = typer.Typer(name="queue", help="Queue management")
contract_app = typer.Typer(name="contract", help="Governance contracts")
trust_app = typer.Typer(name="trust", help="Trust management")
privacy_app = typer.Typer(name="privacy", help="Privacy tools")
audit_app = typer.Typer(name="audit", help="Audit utilities")
auth_app = typer.Typer(name="auth", help="Authorization utilities")
role_app = typer.Typer(name="role", help="Role utilities")
task_app = typer.Typer(name="task", help="Task utilities")
training_app = typer.Typer(name="training", help="Training management")
coach_app = typer.Typer(name="coach", help="Coach utilities")
track_app = typer.Typer(name="track", help="Training track info")
mission_app = typer.Typer(name="mission", help="Mission management")
rep_app = typer.Typer(name="rep", help="Reputation utilities")
delegate_app = typer.Typer(name="delegate", help="Delegation management")
feedback_app = typer.Typer(name="feedback", help="Feedback utilities")
openhands_app = typer.Typer(name="openhands", help="OpenHands agent utilities")
app.add_typer(agent_app)
app.add_typer(team_app)
app.add_typer(model_app)
app.add_typer(config_app)
app.add_typer(session_app)
app.add_typer(queue_app)
app.add_typer(contract_app)
app.add_typer(trust_app)
app.add_typer(privacy_app)
app.add_typer(audit_app)
app.add_typer(auth_app)
app.add_typer(role_app)
app.add_typer(task_app)
app.add_typer(training_app)
app.add_typer(coach_app)
app.add_typer(track_app)
app.add_typer(mission_app)
app.add_typer(rep_app)
app.add_typer(delegate_app)
app.add_typer(feedback_app)
app.add_typer(openhands_app)

agent_app.add_typer(agent_contract_app)


@app.callback(invoke_without_command=True)
def version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
    token: str = typer.Option(None, "--token", help="API token"),
):
    """Global options."""
    if version:
        typer.echo(__version__)
        ctx.exit()
    if token:
        os.environ["AGENTNN_API_TOKEN"] = token


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
        _handle_http_error(err)
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


@session_app.command("budget")
def session_budget(session_id: str) -> None:
    """Show consumed tokens for a session."""
    client = AgentClient()
    data = client.get_session_history(session_id)
    tokens = 0
    for ctx in data.get("context", []):
        tokens += int(ctx.get("metrics", {}).get("tokens_used", 0))
    typer.echo(json.dumps({"session_id": session_id, "tokens_used": tokens}))


@agent_app.command("list")
def agents() -> None:
    """List available agents."""
    client = AgentClient()
    try:
        result = client.list_agents()
    except httpx.HTTPStatusError as err:
        _handle_http_error(err)
        return
    typer.echo(json.dumps(result, indent=2))


@app.command("agents")
def agents_root() -> None:
    """List available agents (alias)."""
    agents()


@agent_app.command("profile")
def agent_profile(name: str) -> None:
    """Show profile information for an agent."""
    client = AgentClient()
    result = client.get_agent_profile(name)
    typer.echo(json.dumps(result, indent=2))


@agent_app.command("update")
def agent_update(name: str, traits: str = "", skills: str = "") -> None:
    """Update an agent profile."""
    client = AgentClient()
    traits_data = json.loads(traits) if traits else None
    skills_list = (
        [s.strip() for s in skills.split(",") if s.strip()] if skills else None
    )
    result = client.update_agent_profile(name, traits=traits_data, skills=skills_list)
    typer.echo(json.dumps(result, indent=2))


@agent_app.command("status")
def agent_status(name: str) -> None:
    """Show live status for an agent."""
    client = AgentClient()
    result = client.get_agent_status(name)
    typer.echo(json.dumps(result, indent=2))


@agent_app.command("top")
def agent_top(
    metric: str = typer.Option("cost", "--metric", help="Sort metric")
) -> None:
    """List agents sorted by metric."""
    client = AgentClient()
    data = client.list_agents().get("agents", [])
    if metric == "skill":
        data.sort(key=lambda a: len(a.get("skills", [])), reverse=True)
    elif metric == "load":
        data.sort(key=lambda a: a.get("load_factor", 0))
    else:
        data.sort(key=lambda a: a.get("estimated_cost_per_token", 0))
    typer.echo(json.dumps(data, indent=2))


@agent_app.command("evolve")
def agent_evolve(name: str, mode: str = "llm", preview: bool = False) -> None:
    """Evolve an agent profile."""
    profile = AgentIdentity.load(name)
    updated = evolve_profile(profile, [], mode)
    if preview:
        typer.echo(json.dumps(asdict(updated), indent=2))
    else:
        updated.save()
        typer.echo(f"profile updated: {name}")


@agent_app.command("keypair")
def agent_keypair(name: str) -> None:
    """Generate or replace a keypair for an agent."""
    generate_keypair(name)
    typer.echo("keypair generated")


@agent_app.command("role")
def agent_role(name: str, set_role: str = typer.Option(None, "--set")) -> None:
    """Show or update the role for an agent."""

    profile = AgentIdentity.load(name)
    if set_role:
        profile.role = set_role
        profile.save()
        typer.echo("updated")
    else:
        typer.echo(profile.role)


@agent_app.command("elevate")
def agent_elevate(name: str, to: str = typer.Option(..., "--to")) -> None:
    """Grant a new role if trust requirements are met."""

    if not eligible_for_role(name, to):
        typer.echo("not eligible")
        raise typer.Exit(code=1)
    contract = AgentContract.load(name)
    if to not in contract.allowed_roles:
        contract.allowed_roles.append(to)
        contract.save()
    typer.echo("elevated")


@agent_app.command("skill")
def agent_skill(name: str) -> None:
    """Show certified skills for an agent."""

    profile = AgentIdentity.load(name)
    typer.echo(json.dumps(profile.certified_skills, indent=2))


@agent_app.command("certify")
def agent_certify(name: str, skill: str = typer.Option(..., "--skill")) -> None:
    """Grant SKILL certification to AGENT."""

    profile = AgentIdentity.load(name)
    skill_def = load_skill(skill)
    if not skill_def:
        typer.echo("skill not found")
        raise typer.Exit(code=1)
    cert = {
        "id": skill_def.id,
        "granted_at": datetime.utcnow().isoformat() + "Z",
        "expires_at": skill_def.expires_at,
    }
    profile.certified_skills = [
        c for c in profile.certified_skills if c.get("id") != skill_def.id
    ]
    profile.certified_skills.append(cert)
    profile.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="certification_granted",
            context_id=name,
            detail={"skill": skill_def.id},
        )
    )
    typer.echo("granted")


@agent_app.command("level")
def agent_level(name: str) -> None:
    """Show level information for AGENT."""

    profile = AgentIdentity.load(name)
    typer.echo(
        json.dumps(
            {
                "current_level": profile.current_level,
                "progress": profile.level_progress,
            },
            indent=2,
        )
    )


@agent_app.command("promote")
def agent_promote(name: str) -> None:
    """Manually trigger level evaluation."""

    profile = AgentIdentity.load(name)
    new_level = check_level_up(profile)
    if new_level:
        typer.echo(f"promoted to {new_level}")
    else:
        typer.echo("no change")


@agent_app.command("rep")
def agent_rep(name: str) -> None:
    """Show reputation information for an agent."""
    profile = AgentIdentity.load(name)
    typer.echo(
        json.dumps(
            {
                "agent": name,
                "reputation": profile.reputation_score,
                "feedback": profile.feedback_log,
            },
            indent=2,
        )
    )


@agent_app.command("endorsements")
def agent_endorsements(name: str) -> None:
    """Show received endorsements for AGENT."""
    from core.trust_network import load_recommendations

    recs = load_recommendations(name)
    typer.echo(json.dumps([asdict(r) for r in recs], indent=2))


@agent_app.command("reflect")
def agent_reflect(name: str) -> None:
    """Analyse feedback and suggest adaptations."""
    from core.feedback_loop import load_feedback
    from core.self_reflection import reflect_and_adapt

    profile = AgentIdentity.load(name)
    feedback = load_feedback(name)
    suggestions = reflect_and_adapt(profile, feedback)
    typer.echo(json.dumps(suggestions, indent=2))


@agent_app.command("adapt")
def agent_adapt(name: str) -> None:
    """Apply recommended adaptations."""
    from core.audit_log import AuditEntry, AuditLog
    from core.feedback_loop import load_feedback
    from core.self_reflection import reflect_and_adapt

    profile = AgentIdentity.load(name)
    feedback = load_feedback(name)
    result = reflect_and_adapt(profile, feedback)
    profile.traits.update(result.get("traits", {}))
    for skill in result.get("skills", []):
        if skill not in profile.skills:
            profile.skills.append(skill)
    profile.adaptation_history.extend(result.get("notes", []))
    profile.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="adaptation_applied",
            context_id=name,
            detail={},
        )
    )
    typer.echo("adapted")


@training_app.command("start")
def training_start(agent: str, path: str = typer.Option(..., "--path")) -> None:
    """Start a training path for an agent."""
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


@training_app.command("progress")
def training_progress(agent: str) -> None:
    """Show training progress for AGENT."""
    profile = AgentIdentity.load(agent)
    typer.echo(json.dumps(profile.training_progress, indent=2))


@mission_app.command("start")
def mission_start(mission_id: str, agent: str = typer.Option(..., "--agent")) -> None:
    """Assign MISSION_ID to AGENT."""
    profile = AgentIdentity.load(agent)
    profile.active_missions.append(mission_id)
    profile.mission_progress[mission_id] = {"step": 0, "status": "in_progress"}
    profile.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="mission_started",
            context_id=agent,
            detail={"mission": mission_id},
        )
    )
    typer.echo("started")


@mission_app.command("progress")
def mission_progress(agent: str) -> None:
    """Show mission progress for AGENT."""
    profile = AgentIdentity.load(agent)
    typer.echo(json.dumps(profile.mission_progress, indent=2))


@mission_app.command("step")
def mission_step(agent: str) -> None:
    """Display next mission step for AGENT."""
    profile = AgentIdentity.load(agent)
    if not profile.active_missions:
        typer.echo("no mission")
        raise typer.Exit(code=1)
    mid = profile.active_missions[0]
    from core.mission_prompts import render_prompt
    from core.missions import AgentMission

    mission = AgentMission.load(mid)
    if not mission:
        typer.echo("missing mission")
        raise typer.Exit(code=1)
    step_idx = profile.mission_progress.get(mid, {}).get("step", 0)
    if step_idx >= len(mission.steps):
        typer.echo("mission complete")
        return
    prompt = render_prompt(profile, mission.steps[step_idx], {"goal": mission.title})
    typer.echo(prompt)


@mission_app.command("complete")
def mission_complete(agent: str) -> None:
    """Complete active mission if finished."""
    from core.missions import AgentMission
    from core.rewards import grant_rewards

    profile = AgentIdentity.load(agent)
    if not profile.active_missions:
        typer.echo("no mission")
        raise typer.Exit(code=1)
    mid = profile.active_missions[0]
    mission = AgentMission.load(mid)
    progress = profile.mission_progress.get(mid, {"step": 0})
    if progress.get("step", 0) >= len(mission.steps):
        grant_rewards(agent, mission.rewards)
        profile.active_missions.remove(mid)
        progress["status"] = "complete"
        profile.mission_progress[mid] = progress
        profile.save()
        log = AuditLog()
        log.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="cli",
                action="mission_completed",
                context_id=agent,
                detail={"mission": mid},
            )
        )
        log.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="cli",
                action="reward_unlocked",
                context_id=agent,
                detail=mission.rewards,
            )
        )
        typer.echo("completed")
    else:
        typer.echo("not finished")


@track_app.command("show")
def track_show() -> None:
    """List available training paths."""

    from core.training import TRAINING_DIR

    paths = []
    for file in TRAINING_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as fh:
            paths.append(json.load(fh))
    typer.echo(json.dumps(paths, indent=2))


@coach_app.command("evaluate")
def coach_evaluate(
    agent: str,
    skill: str = typer.Option(..., "--skill"),
    score: float = typer.Option(1.0, "--score"),
) -> None:
    """Submit evaluation result for a skill."""
    profile = AgentIdentity.load(agent)
    profile.training_log.append(
        {
            "skill": skill,
            "evaluation_score": score,
            "last_attempted": datetime.utcnow().isoformat(),
        }
    )
    if score >= 0.8:
        profile.training_progress[skill] = "complete"
        auto_certify(agent, skill)
    profile.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="evaluation_submitted",
            context_id=agent,
            detail={"skill": skill, "score": score},
        )
    )
    typer.echo("recorded")


@role_app.command("limits")
def role_limits(role: str) -> None:
    """Show resource limits for ROLE."""
    from core.role_capabilities import ROLE_CAPABILITIES

    typer.echo(json.dumps({role: ROLE_CAPABILITIES.get(role, {})}, indent=2))


@task_app.command("limits")
def task_limits(context_file: str, role: str = typer.Option(..., "--role")) -> None:
    """Preview applied limits for CONTEXT_FILE and ROLE."""
    from core.role_capabilities import apply_role_capabilities

    with open(context_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "task_type" in data:
        ctx = ModelContext(task_context=TaskContext(**data))
    else:
        ctx = ModelContext(**data)
    apply_role_capabilities(ctx, role)
    typer.echo(json.dumps(ctx.applied_limits, indent=2))


@team_app.command("create")
def team_create(
    name: str,
    coordinator: str = typer.Option(None, "--coordinator"),
) -> None:
    """Create a new team."""
    from dataclasses import asdict
    from datetime import datetime
    from uuid import uuid4

    from core.teams import AgentTeam

    team_id = str(uuid4())
    team = AgentTeam(
        id=team_id,
        name=name,
        members=[coordinator] if coordinator else [],
        shared_goal=None,
        skills_focus=[],
        coordinator=coordinator,
        created_at=datetime.utcnow().isoformat(),
    )
    team.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="team_created",
            context_id=team_id,
            detail={"name": name},
        )
    )
    typer.echo(json.dumps(asdict(team), indent=2))


@team_app.command("join")
def team_join(
    team_id: str,
    agent: str = typer.Option(..., "--agent"),
    role: str = typer.Option("peer", "--role"),
) -> None:
    """Join an existing team."""
    from core.teams import AgentTeam

    profile = AgentIdentity.load(agent)
    profile.team_id = team_id
    profile.team_role = role
    profile.save()
    team = AgentTeam.load(team_id)
    if agent not in team.members:
        team.members.append(agent)
        team.save()
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="cli",
            action="joined_team",
            context_id=team_id,
            detail={"agent": agent, "role": role},
        )
    )
    typer.echo("joined")


@team_app.command("show")
def team_show(team_id: str) -> None:
    """Show team information."""
    from dataclasses import asdict

    from core.teams import AgentTeam

    team = AgentTeam.load(team_id)
    typer.echo(json.dumps(asdict(team), indent=2))


@team_app.command("share-skill")
def team_share_skill(
    agent: str,
    skill: str = typer.Option(..., "--skill"),
) -> None:
    """Share a skill insight with the agent's team."""
    from core.team_knowledge import broadcast_insight

    broadcast_insight(agent, skill, {"event": "share"})
    typer.echo("shared")


@config_app.command("show")
def config_show() -> None:
    """Show effective configuration."""
    settings = SDKSettings.load()
    typer.echo(json.dumps(settings.__dict__, indent=2))


@config_app.command("check")
def config_check() -> None:
    """Validate and display core configuration."""
    from core.config import settings as core_settings

    typer.echo(json.dumps(core_settings.model_dump(), indent=2))


@model_app.command("list")
def model_list():
    """List available models."""
    client = AgentClient()
    models = client.get_models()
    typer.echo(json.dumps(models, indent=2))


@model_app.command("runs-view")
def model_runs_view(run_id: str):
    """Show details for a run."""
    mgr = ModelManager()
    info = mgr.get_run_summary(run_id)
    typer.echo(json.dumps(info, indent=2))


@model_app.command("register")
def model_register(name: str, run_id: str):
    """Register a model from a run."""
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    client.create_registered_model(name)
    client.create_model_version(name, model_uri, run_id)
    typer.echo("registered")


@model_app.command("stage")
def model_stage(name: str, stage: str):
    """Transition the latest version of a model."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        typer.echo("no versions")
        return
    version = versions[0].version
    client.transition_model_version_stage(name, version, stage)
    typer.echo(f"{name} -> {stage}")


@model_app.command("switch")
def model_switch(model_id: str, user_id: str = "default"):
    """Switch active model."""
    client = AgentClient()
    resp = client.set_model(model_id, user_id)
    typer.echo(json.dumps(resp))


@queue_app.command("status")
def queue_status() -> None:
    """Show current dispatch queue status."""
    client = AgentClient()
    resp = client._client.get("/queue/status")
    resp.raise_for_status()
    typer.echo(resp.text)


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
    from core.trust_network import AgentRecommendation, record_recommendation

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
    from core.trust_circle import is_trusted_for
    from core.trust_network import load_recommendations

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

    allowed = is_authorized(agent, role, action, resource)
    typer.echo(json.dumps({"authorized": allowed}))


@agent_contract_app.command("set-access")
def set_access_level(agent: str, max_level: AccessLevel) -> None:
    """Update max access level for an agent contract."""
    contract = AgentContract.load(agent)
    contract.max_access_level = max_level
    contract.save()
    typer.echo("updated")


@app.command("promote")
def task_promote(task_id: str) -> None:
    """Promote a queued task by id."""
    client = AgentClient()
    resp = client._client.post(f"/queue/promote/{task_id}")
    resp.raise_for_status()
    typer.echo(resp.text)


@rep_app.command("leaderboard")
def rep_leaderboard() -> None:
    """Show top agents by reputation."""
    results = []
    for file in PROFILE_DIR.glob("*.json"):
        name = file.stem
        score = aggregate_score(name)
        results.append({"agent": name, "score": round(score, 3)})
    results.sort(key=lambda x: x["score"], reverse=True)
    typer.echo(json.dumps(results, indent=2))


@delegate_app.command("grant")
def delegate_grant(
    from_agent: str,
    to_agent: str,
    role: str = typer.Option(..., "--role"),
    scope: str = typer.Option("task", "--scope"),
    reason: str = typer.Option(None, "--reason"),
    expires_at: str = typer.Option(None, "--expires-at"),
) -> None:
    """Grant delegation from FROM_AGENT to TO_AGENT."""
    from core.delegation import grant_delegation

    grant_delegation(from_agent, to_agent, role, scope, expires_at, reason)
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor=from_agent,
            action="delegation_granted",
            context_id=to_agent,
            detail={"role": role, "scope": scope},
        )
    )
    typer.echo("granted")


@delegate_app.command("revoke")
def delegate_revoke(
    from_agent: str, to_agent: str, role: str = typer.Option(..., "--role")
) -> None:
    """Revoke delegation."""
    from core.delegation import revoke_grant

    revoke_grant(from_agent, to_agent, role)
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor=from_agent,
            action="delegation_revoked",
            context_id=to_agent,
            detail={"role": role},
        )
    )
    typer.echo("revoked")


@delegate_app.command("list")
def delegate_list(agent: str) -> None:
    """List delegations issued by AGENT."""
    from core.delegation import load_grants

    grants = load_grants(agent)
    typer.echo(json.dumps([asdict(g) for g in grants], indent=2))


@feedback_app.command("submit")
def feedback_submit(
    session: str,
    score: int = typer.Option(..., "--score"),
    comment: str = typer.Option("", "--comment"),
    agent: str = typer.Option(None, "--agent"),
    user: str = typer.Option("default", "--user"),
) -> None:
    """Send feedback for a session entry."""
    client = AgentClient()
    payload = {
        "session_id": session,
        "user_id": user,
        "agent_id": agent or "",
        "score": score,
        "comment": comment or None,
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        resp = client.post_feedback(session, payload)
    except httpx.HTTPStatusError as err:
        _handle_http_error(err)
        return
    typer.echo(json.dumps(resp, indent=2))


@feedback_app.command("log")
def feedback_log(agent: str) -> None:
    """Show feedback entries for AGENT."""
    from core.feedback_loop import load_feedback

    entries = load_feedback(agent)
    typer.echo(json.dumps([asdict(e) for e in entries], indent=2))


@openhands_app.command("list")
def openhands_list() -> None:
    """List registered OpenHands agents."""
    ports_env = os.getenv("OPENHANDS_AGENT_PORTS", "3001-3010")
    if "-" in ports_env:
        start, end = [int(p) for p in ports_env.split("-")]
        ports = list(range(start, end + 1))
    else:
        ports = [int(p) for p in ports_env.split(",") if p]
    agents = [
        {"name": f"openhands_{i}", "url": f"http://localhost:{p}", "port": p}
        for i, p in enumerate(ports, start=1)
    ]
    typer.echo(json.dumps({"agents": agents}, indent=2))


@openhands_app.command("trigger")
def openhands_trigger(task: str, agents: str = "all") -> None:
    """Trigger a task on one or multiple OpenHands agents."""
    ports_env = os.getenv("OPENHANDS_AGENT_PORTS", "3001-3010")
    if "-" in ports_env:
        start, end = [int(p) for p in ports_env.split("-")]
        ports = list(range(start, end + 1))
    else:
        ports = [int(p) for p in ports_env.split(",") if p]
    selected = (
        ports if agents == "all" else [ports[int(a) - 1] for a in agents.split(",")]
    )
    results = {}
    for port in selected:
        try:
            resp = httpx.post(
                f"http://localhost:{port}/api/conversations",
                json={"initial_user_msg": task},
                timeout=5,
            )
            resp.raise_for_status()
            results[port] = resp.json()
        except Exception as exc:  # pragma: no cover - network
            results[port] = {"error": str(exc)}
    typer.echo(json.dumps(results, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""Command line tool for Agent-NN."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import mlflow
import typer

from core.access_control import is_authorized
from core.agent_evolution import evolve_profile
from core.agent_profile import AgentIdentity
from core.audit_log import AuditLog
from core.crypto import generate_keypair, verify_signature
from core.governance import AgentContract
from core.model_context import ModelContext, TaskContext
from core.privacy import AccessLevel
from core.privacy_filter import redact_context
from core.trust_evaluator import calculate_trust, eligible_for_role

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

agent_app.add_typer(agent_contract_app)


@app.callback(invoke_without_command=True)
def version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
):
    """Global options."""
    if version:
        typer.echo(__version__)
        ctx.exit()


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
def sessions() -> None:
    """List active sessions."""
    client = AgentClient()
    result = client.list_sessions()
    typer.echo(json.dumps(result, indent=2))


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
    result = client.list_agents()
    typer.echo(json.dumps(result, indent=2))


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
    goal: str = typer.Option(..., "--goal"),
    leader: str = "",
    members: str = "",
    strategy: str = "plan-then-split",
) -> None:
    """Create a new agent coalition."""
    client = AgentClient()
    member_list = [m.strip() for m in members.split(",") if m.strip()]
    result = client.create_coalition(goal, leader, member_list, strategy)
    typer.echo(json.dumps(result, indent=2))


@team_app.command("assign")
def team_assign(
    coalition_id: str,
    to: str = typer.Option(..., "--to"),
    task: str = typer.Option(..., "--task"),
) -> None:
    """Assign a subtask to a member."""
    client = AgentClient()
    result = client.assign_subtask(coalition_id, to, task)
    typer.echo(json.dumps(result, indent=2))


@team_app.command("status")
def team_status(coalition_id: str) -> None:
    """Show coalition status."""
    client = AgentClient()
    result = client.get_coalition(coalition_id)
    typer.echo(json.dumps(result, indent=2))


@config_app.command("show")
def config_show() -> None:
    """Show effective configuration."""
    settings = SDKSettings.load()
    typer.echo(json.dumps(settings.__dict__, indent=2))


@model_app.command("list")
def model_list():
    """List MLflow experiments."""
    mgr = ModelManager()
    experiments = mgr.list_experiments()
    typer.echo(json.dumps(experiments))


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

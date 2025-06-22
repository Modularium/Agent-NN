"""Command line tool for Agent-NN."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import os

import mlflow
import typer

from core.agent_evolution import evolve_profile
from core.agent_profile import AgentIdentity
from core.governance import AgentContract
from core.trust_evaluator import calculate_trust
from core.model_context import ModelContext, TaskContext
from core.privacy_filter import redact_context
from core.privacy import AccessLevel

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
app.add_typer(agent_app)
app.add_typer(team_app)
app.add_typer(model_app)
app.add_typer(config_app)
app.add_typer(session_app)
app.add_typer(queue_app)
app.add_typer(contract_app)
app.add_typer(trust_app)
app.add_typer(privacy_app)

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


@trust_app.command("score")
def trust_score(agent: str) -> None:
    """Calculate trust score for an agent."""
    score = calculate_trust(agent, [])
    typer.echo(json.dumps({"agent": agent, "score": score}, indent=2))


@privacy_app.command("preview")
def privacy_preview(agent: str, task_file: str) -> None:
    """Show context after privacy filtering for an agent."""
    with open(task_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ctx = ModelContext(task_context=TaskContext(**data))
    contract = AgentContract.load(agent)
    filtered = redact_context(ctx, contract.max_access_level)
    typer.echo(json.dumps(filtered.model_dump(), indent=2))


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

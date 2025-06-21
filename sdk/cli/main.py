"""Command line tool for Agent-NN."""
from __future__ import annotations

import json
import typer
import mlflow
from .. import __version__

from ..client import AgentClient
from ..config import SDKSettings
from ..nn_models import ModelManager
from core.agent_profile import AgentIdentity
from core.agent_evolution import evolve_profile
from dataclasses import asdict


model_app = typer.Typer(name="model", help="Model management commands")
config_app = typer.Typer(name="config", help="Configuration commands")

app = typer.Typer()
agent_app = typer.Typer(name="agent", help="Agent management")
app.add_typer(agent_app)
app.add_typer(model_app)
app.add_typer(config_app)

@app.callback(invoke_without_command=True)
def version_callback(ctx: typer.Context,
                     version: bool = typer.Option(False, '--version', help='Show version and exit')):
    """Global options."""
    if version:
        typer.echo(__version__)
        ctx.exit()


@app.command()
def submit(task: str) -> None:
    """Submit a task to the dispatcher."""
    client = AgentClient()
    result = client.submit_task(task)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def sessions() -> None:
    """List active sessions."""
    client = AgentClient()
    result = client.list_sessions()
    typer.echo(json.dumps(result, indent=2))


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
    skills_list = [s.strip() for s in skills.split(",") if s.strip()] if skills else None
    result = client.update_agent_profile(name, traits=traits_data, skills=skills_list)
    typer.echo(json.dumps(result, indent=2))


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

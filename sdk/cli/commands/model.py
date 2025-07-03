"""Model management commands."""
from __future__ import annotations

import json

import typer
import mlflow

from ..client import AgentClient
from ..nn_models import ModelManager
from ..utils.formatting import print_output

model_app = typer.Typer(name="model", help="Model management commands")


@model_app.command("list")
def model_list(
    output: str = typer.Option("table", "--output", help="table|json|markdown")
) -> None:
    """List available models."""
    client = AgentClient()
    models = client.get_models().get("models", [])
    print_output(models, output)


@model_app.command("runs-view")
def model_runs_view(run_id: str) -> None:
    """Show details for a run."""
    mgr = ModelManager()
    info = mgr.get_run_summary(run_id)
    typer.echo(json.dumps(info, indent=2))


@model_app.command("register")
def model_register(name: str, run_id: str) -> None:
    """Register a model from a run."""
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    client.create_registered_model(name)
    client.create_model_version(name, model_uri, run_id)
    typer.echo("registered")


@model_app.command("stage")
def model_stage(name: str, stage: str) -> None:
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
def model_switch(model_id: str, user_id: str = "default") -> None:
    """Switch active model."""
    client = AgentClient()
    resp = client.set_model(model_id, user_id)
    typer.echo(json.dumps(resp))


def register(app: typer.Typer) -> None:
    app.add_typer(model_app)

__all__ = ["register", "model_app"]

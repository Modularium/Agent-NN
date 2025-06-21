"""SDK interface for MLflow based model tracking."""
from __future__ import annotations

from typing import Dict, Any, List
import uuid
import mlflow

from ..config import SDKSettings
from ..client.agent_client import AgentClient
from mlflow_integration.client import MLflowClientWrapper


class ModelManager:
    """High level MLflow integration for developers."""

    def __init__(self, settings: SDKSettings | None = None) -> None:
        self.settings = settings or SDKSettings.load()
        self.mlflow = MLflowClientWrapper()
        self.client = AgentClient(self.settings)

    def track_run(self, model_name: str, config: Dict[str, Any], metrics: Dict[str, float], tags: Dict[str, str]) -> str:
        """Create an MLflow run with the given data."""
        run_name = tags.get("run_name") or f"{model_name}-{uuid.uuid4().hex[:8]}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("model_name", model_name)
            for k, v in config.items():
                mlflow.log_param(k, v)
            for mk, mv in metrics.items():
                mlflow.log_metric(mk, mv)
            for tk, tv in tags.items():
                mlflow.set_tag(tk, tv)
            return run.info.run_id

    def list_experiments(self) -> List[str]:
        """Return all experiment names from the tracking server."""
        return self.mlflow.list_experiments()

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Return parameters and metrics for a run."""
        return self.mlflow.get_run_summary(run_id)

    def load_model_from_registry(self, name: str, stage: str):
        """Load a model from the MLflow registry."""
        model_uri = f"models:/{name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)

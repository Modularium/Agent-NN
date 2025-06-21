"""Thin wrapper around the mlflow client API used by the SDK."""
from __future__ import annotations

from typing import Dict, Any
import mlflow
from config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class MLflowClientWrapper:
    """Helper to interact with the MLflow tracking server."""

    def __init__(self) -> None:
        self.client = mlflow.tracking.MlflowClient()

    def list_experiments(self) -> list[str]:
        if hasattr(self.client, "list_experiments"):
            return [exp.name for exp in self.client.list_experiments()]
        return []

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        data = self.client.get_run(run_id)
        return {
            "run_id": data.info.run_id,
            "status": data.info.status,
            "metrics": data.data.metrics,
            "params": data.data.params,
        }

    def register_model(self, model_uri: str, name: str) -> None:
        self.client.create_registered_model(name)
        self.client.create_model_version(name, model_uri, run_id=None)

    def transition_model(self, name: str, version: str, stage: str) -> None:
        self.client.transition_model_version_stage(name, version, stage)

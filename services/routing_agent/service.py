"""Route tasks to workers using rules and optional MetaLearner."""
from __future__ import annotations

import os
from typing import Dict, List

import yaml

from .config import settings
from core.metrics_utils import ROUTING_DECISIONS

try:
    from nn_models.meta_learner import MetaLearner
except Exception:  # pragma: no cover - optional dependency
    MetaLearner = None


class RoutingAgentService:
    """Route tasks based on YAML rules and optional model prediction."""

    def __init__(self, rules_path: str | None = None) -> None:
        path = rules_path or settings.rules_path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.rules: Dict[str, str] = yaml.safe_load(fh) or {}
        else:
            self.rules = {}
        self.meta = MetaLearner() if settings.meta_enabled and MetaLearner else None

    def predict_agent(self, ctx: dict) -> str:
        """Return target worker for context using rules and meta model."""
        ttype = ctx.get("task_type")
        target = self.rules.get(ttype)
        if target:
            return target
        if self.meta:
            try:
                pred = self.meta.predict_agent(ctx)
                if pred:
                    return pred
            except Exception:  # pragma: no cover - prediction failure
                pass
        return "worker_dev"

    def route(
        self,
        task_type: str,
        required_tools: List[str] | None = None,
        context: Dict | None = None,
    ) -> Dict:
        ctx = {"task_type": task_type, "required_tools": required_tools}
        if context:
            ctx.update(context)
        worker = self.predict_agent(ctx)
        ROUTING_DECISIONS.labels(task_type, worker).inc()
        return {"target_worker": worker}

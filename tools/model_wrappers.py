"""Simple wrappers exposing nn_models via a unified interface."""

from __future__ import annotations

from typing import Any, Dict


def _torch_available() -> bool:
    try:  # pragma: no cover - optional dependency
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception:  # pragma: no cover - importlib failure
        return False


class BaseModelTool:
    """Base interface for model tools."""

    description: str = ""
    parameters: Dict[str, Any] = {}
    input_schema: Dict[str, Any] = {}

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MultiTaskModelWrapper(BaseModelTool):
    """Wrapper for ``nn_models.multi_task_learning``."""

    description = "Multi-task learning model"

    def run(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - optional torch
        if not _torch_available():
            return {"error": "torch not installed"}
        import torch

        from nn_models.multi_task_learning import TaskEncoder

        model = TaskEncoder(input_dim=4, hidden_dims=[4], output_dim=2)
        x = torch.zeros(1, 4)
        with torch.no_grad():
            out = model(x)
        return {"output": out.tolist()}


class AgentV2Wrapper(BaseModelTool):
    """Wrapper for ``nn_models.agent_nn_v2``."""

    description = "AgentNN v2 model"

    def run(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - optional torch
        if not _torch_available():
            return {"error": "torch not installed"}
        import torch

        from nn_models.agent_nn_v2 import AgentNN

        model = AgentNN()
        x = torch.zeros(1, 768)
        with torch.no_grad():
            out = model(x)
        return {"output": out.tolist()}


class DynamicArchitectureWrapper(BaseModelTool):
    """Wrapper for ``nn_models.dynamic_architecture``."""

    description = "Dynamic architecture network"

    def run(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - optional torch
        if not _torch_available():
            return {"error": "torch not installed"}
        import torch

        from nn_models.dynamic_architecture import DynamicArchitecture

        model = DynamicArchitecture(input_dim=4, output_dim=2)
        x = torch.zeros(1, 4)
        with torch.no_grad():
            out = model(x)
        return {"output": out.tolist()}


class AgentNNWrapper(BaseModelTool):
    """Wrapper for ``nn_models.agent_nn``."""

    description = "AgentNN base model"

    def run(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - optional torch
        if not _torch_available():
            return {"error": "torch not installed"}
        import torch

        from nn_models.agent_nn import AgentNN

        model = AgentNN()
        x = torch.zeros(1, 768)
        with torch.no_grad():
            out = model(x)
        return {"output": out.tolist()}

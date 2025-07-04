from __future__ import annotations

"""Shared service helpers for CLI and API."""

import asyncio
from typing import Any, Dict

from agentnn.deployment.agent_registry import AgentRegistry
from core.model_context import ModelContext
from managers.agent_optimizer import AgentOptimizer
from managers.model_manager import ModelManager
from sdk.client import AgentClient

__all__ = [
    "create_agent",
    "dispatch_task",
    "evaluate_agent",
    "load_model",
    "train_model",
]


def create_agent(
    config: Dict[str, Any], endpoint: str = "http://localhost:8090"
) -> Dict[str, Any]:
    """Register ``config`` with the MCP agent registry."""
    registry = AgentRegistry(endpoint)
    return registry.deploy(config)


def dispatch_task(ctx: ModelContext) -> Dict[str, Any]:
    """Send ``ctx`` to the dispatcher and return the result."""
    client = AgentClient()
    return client.dispatch_task(ctx)


def evaluate_agent(agent_id: str) -> Dict[str, Any]:
    """Return evaluation metrics for ``agent_id``."""
    optimizer = AgentOptimizer()
    return asyncio.run(optimizer.evaluate_agent(agent_id))


def load_model(
    name: str,
    type: str,
    source: str,
    config: Dict[str, Any],
    version: str | None = None,
) -> Dict[str, Any]:
    """Load a model using :class:`ModelManager`."""
    manager = ModelManager()
    return asyncio.run(manager.load_model(name, type, source, config, version))


def train_model(args: Any) -> Any:
    """Run the training routine with ``args``."""
    from training.train import train

    return train(args)

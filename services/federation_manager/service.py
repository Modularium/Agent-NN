"""Manage federated Agent-NN nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import httpx

from core.model_context import ModelContext


@dataclass
class FederatedNode:
    """Information about a remote Agent-NN instance."""

    name: str
    base_url: str


class FederationManagerService:
    """Register and dispatch tasks to federated nodes."""

    def __init__(self) -> None:
        self.nodes: Dict[str, FederatedNode] = {}

    def register_node(self, name: str, base_url: str) -> None:
        self.nodes[name] = FederatedNode(name, base_url.rstrip("/"))

    def remove_node(self, name: str) -> None:
        self.nodes.pop(name, None)

    def dispatch(self, node_name: str, ctx: ModelContext) -> ModelContext:
        node = self.nodes.get(node_name)
        if not node:
            raise ValueError(f"node {node_name} not registered")
        with httpx.Client() as client:
            resp = client.post(
                f"{node.base_url}/dispatch",
                json=ctx.model_dump(),
                timeout=10,
            )
            resp.raise_for_status()
            return ModelContext(**resp.json())

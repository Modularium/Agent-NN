from __future__ import annotations
"""Federated learning utilities."""

from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Dict  # noqa: E402

import torch  # noqa: E402


@dataclass
class ClientUpdate:
    """Weights and metadata from a single client."""

    client_id: str
    weights: Dict[str, torch.Tensor]
    samples: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class FederatedAveraging:
    """Aggregate model updates from multiple clients."""

    def __init__(self) -> None:
        self._updates: list[ClientUpdate] = []

    def add_update(
        self, client_id: str, state: Dict[str, torch.Tensor], samples: int
    ) -> None:
        self._updates.append(ClientUpdate(client_id, state, samples))

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self._updates:
            raise ValueError("no updates to aggregate")
        total_samples = sum(u.samples for u in self._updates)
        agg: Dict[str, torch.Tensor] = {}
        for update in self._updates:
            for k, v in update.weights.items():
                if k not in agg:
                    agg[k] = v.clone() * (update.samples / total_samples)
                else:
                    agg[k] += v * (update.samples / total_samples)
        self._updates.clear()
        return agg

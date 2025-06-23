from __future__ import annotations

import logging
from typing import Dict

from services.session_manager.service import SessionManagerService
from .training.weights import accumulate_weights

logger = logging.getLogger(__name__)


class AutoTrainer:
    """Simple feedback-driven trainer."""

    def __init__(self, service: SessionManagerService) -> None:
        self.service = service
        self.weights: Dict[str, float] = {}

    def run(self) -> None:
        """Analyse feedback and adjust weights."""
        entries = []
        for fb in self.service.feedback_store.all_feedback():
            ctxs = self.service.get_context(fb.session_id)
            for ctx in ctxs:
                if (
                    ctx.agent_selection == fb.agent_id
                    and ctx.task_context
                    and ctx.task_context.task_type == "docker"
                ):
                    entries.append((fb.agent_id, fb.score))
        stats = accumulate_weights(entries)
        if stats:
            logger.info("auto_trainer_update", weights=stats)
            for agent, weight in stats.items():
                self.weights[agent] = self.weights.get(agent, 0.0) + weight


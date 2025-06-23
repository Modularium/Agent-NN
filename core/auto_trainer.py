from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

from services.session_manager.service import SessionManagerService

logger = logging.getLogger(__name__)


class AutoTrainer:
    """Simple feedback-driven trainer."""

    def __init__(self, service: SessionManagerService) -> None:
        self.service = service
        self.weights: Dict[str, float] = {}

    def run(self) -> None:
        """Analyse feedback and adjust weights."""
        stats: Dict[str, float] = defaultdict(float)
        for fb in self.service.feedback_store.all_feedback():
            ctxs = self.service.get_context(fb.session_id)
            for ctx in ctxs:
                if (
                    ctx.agent_selection == fb.agent_id
                    and ctx.task_context
                    and ctx.task_context.task_type == "docker"
                ):
                    stats[fb.agent_id] += fb.score
        if stats:
            logger.info("auto_trainer_update", weights=stats)
        self.weights.update({k: self.weights.get(k, 0.0) + v for k, v in stats.items()})


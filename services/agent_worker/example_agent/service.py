"""Example worker implementation."""

from typing import Dict

from core.metrics_utils import TASKS_PROCESSED


class ExampleAgentService:
    """Simple agent worker echoing tasks."""

    def execute_task(self, task: str) -> Dict[str, str]:
        """Return a dummy result."""
        TASKS_PROCESSED.labels("example_agent").inc()
        return {"result": f"processed {task}"}

"""Example worker implementation."""

from typing import Dict


class ExampleAgentService:
    """Simple agent worker echoing tasks."""

    def execute_task(self, task: str) -> Dict[str, str]:
        """Return a dummy result."""
        return {"result": f"processed {task}"}

"""Task dispatcher core logic."""

from core.model_context import TaskContext


class TaskDispatcherService:
    """Dispatch incoming tasks to worker agents."""

    def dispatch_task(self, task: TaskContext) -> dict:
        """Stub dispatch method returning the received payload."""
        return {"task_id": task.task_id, "status": "queued"}

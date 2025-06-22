import pytest

from core.model_context import AgentRunContext, TaskContext
from services.task_dispatcher.service import TaskDispatcherService


@pytest.mark.integration
def test_queue_execution_flow(monkeypatch):
    service = TaskDispatcherService()
    monkeypatch.setattr(
        service,
        "_fetch_agents",
        lambda cap: [{"id": "a1", "url": "http://a", "name": "a1"}],
    )
    monkeypatch.setattr(
        service,
        "_run_agent",
        lambda a, c: AgentRunContext(agent_id=a["id"], result="ok"),
    )
    ctx = service.enqueue_task(TaskContext(task_type="demo"))
    assert ctx.dispatch_state == "queued"
    result = service.process_queue_once()
    assert result.dispatch_state == "completed"
    assert result.result == "ok"

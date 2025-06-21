from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, ModelContext


def test_voting_mode(monkeypatch):
    service = TaskDispatcherService()
    agents = [
        {"id": "w1", "url": "http://writer", "capabilities": ["demo"], "role": "writer"},
        {"id": "c1", "url": "http://critic", "capabilities": ["demo"], "role": "critic"},
    ]
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: agents)

    def fake_coord(ctx: ModelContext, mode: str):
        ctx.agents[0].result = "answer"
        ctx.agents[0].score = 0.8
        ctx.agents[1].result = None
        ctx.aggregated_result = "answer"
        return ctx

    monkeypatch.setattr(service, "_send_to_coordinator", fake_coord)

    ctx = service.dispatch_task(TaskContext(task_type="demo"), mode="voting")
    assert ctx.aggregated_result == "answer"
    assert ctx.agents[0].score == 0.8

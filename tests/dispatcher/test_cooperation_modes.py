from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, ModelContext, AgentRunContext


def test_parallel_mode(monkeypatch):
    service = TaskDispatcherService()
    agents = [
        {"id": "r1", "url": "http://retriever", "capabilities": ["demo"], "role": "retriever"},
        {"id": "w1", "url": "http://writer", "capabilities": ["demo"], "role": "writer"},
    ]
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: agents)

    def fake_coord(ctx: ModelContext, mode: str):
        ctx.agents[0].result = [{"text": "doc"}]
        ctx.agents[1].result = "written"
        ctx.aggregated_result = [a.result for a in ctx.agents]
        return ctx

    monkeypatch.setattr(service, "_send_to_coordinator", fake_coord)

    ctx = service.dispatch_task(TaskContext(task_type="demo"), mode="parallel")
    assert ctx.aggregated_result == [[{"text": "doc"}], "written"]
    assert len(ctx.agents) == 2


def test_orchestrated_mode(monkeypatch):
    service = TaskDispatcherService()
    agents = [
        {"id": "r1", "url": "http://retriever", "capabilities": ["demo"], "role": "retriever"},
        {"id": "w1", "url": "http://writer", "capabilities": ["demo"], "role": "writer"},
    ]
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: agents)

    def fake_coord(ctx: ModelContext, mode: str):
        ctx.agents[0].result = [{"text": "doc"}]
        ctx.agents[1].result = "final"
        ctx.aggregated_result = "final"
        return ctx

    monkeypatch.setattr(service, "_send_to_coordinator", fake_coord)

    ctx = service.dispatch_task(TaskContext(task_type="demo"), mode="orchestrated")
    assert ctx.aggregated_result == "final"
    assert len(ctx.agents) == 2

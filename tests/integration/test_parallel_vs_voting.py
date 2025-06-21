from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, ModelContext


def test_parallel_vs_voting(monkeypatch):
    service = TaskDispatcherService()
    agents = [
        {"id": "w1", "url": "http://writer", "capabilities": ["demo"], "role": "writer"},
        {"id": "c1", "url": "http://critic", "capabilities": ["demo"], "role": "critic"},
    ]
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: agents)

    def fake_coord(ctx: ModelContext, mode: str):
        if mode == "parallel":
            ctx.agents[0].result = "text"
            ctx.agents[1].result = None
            ctx.aggregated_result = ["text", None]
        else:
            ctx.agents[0].result = "text"
            ctx.agents[0].score = 0.9
            ctx.agents[1].result = None
            ctx.aggregated_result = "text"
        return ctx

    monkeypatch.setattr(service, "_send_to_coordinator", fake_coord)

    parallel = service.dispatch_task(TaskContext(task_type="demo"), mode="parallel")
    voting = service.dispatch_task(TaskContext(task_type="demo"), mode="voting")

    assert parallel.aggregated_result != voting.aggregated_result
    assert voting.agents[0].score == 0.9

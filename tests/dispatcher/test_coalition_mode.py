from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, ModelContext


def test_coalition_mode(monkeypatch):
    service = TaskDispatcherService()
    agents = [{"id": "w1", "url": "http://writer", "capabilities": ["demo"], "role": "writer"}]
    monkeypatch.setattr(service, "_fetch_agents", lambda cap: agents)
    monkeypatch.setattr(service, "_init_coalition", lambda goal, members: {"id": "c1"})
    monkeypatch.setattr(service, "_assign_subtask", lambda cid, title, mem: None)

    def fake_coord(ctx: ModelContext, mode: str):
        ctx.agents[0].result = "answer"
        ctx.agents[0].subtask_result = "answer"
        ctx.aggregated_result = [a.result for a in ctx.agents]
        return ctx

    monkeypatch.setattr(service, "_send_to_coordinator", fake_coord)

    ctx = service.dispatch_task(TaskContext(task_type="demo"), mode="coalition")
    assert ctx.aggregated_result == ["answer"]
    assert ctx.task_context.preferences["coalition_id"] == "c1"

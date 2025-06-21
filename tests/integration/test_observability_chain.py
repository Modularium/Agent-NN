from core.model_context import TaskContext
from services.task_dispatcher.service import TaskDispatcherService
from core.metrics_utils import TASKS_PROCESSED


def test_observability_chain(monkeypatch):
    dispatcher = TaskDispatcherService()
    monkeypatch.setattr(dispatcher, '_fetch_agents', lambda cap: [])
    before = TASKS_PROCESSED.labels('task_dispatcher')._value.get()
    ctx = dispatcher.dispatch_task(TaskContext(task_type='demo'))
    after = TASKS_PROCESSED.labels('task_dispatcher')._value.get()
    assert after == before + 1
    assert ctx.task == ctx.task_context.task_id

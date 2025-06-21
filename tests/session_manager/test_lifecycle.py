from services.session_manager.service import SessionManagerService
from core.model_context import ModelContext, TaskContext


def test_session_lifecycle():
    service = SessionManagerService()
    sid = service.start_session()
    ctx = ModelContext(task_context=TaskContext(task_type="demo"), session_id=sid)
    service.update_context(ctx)
    history = service.get_context(sid)
    assert history[-1].task_context.task_type == "demo"

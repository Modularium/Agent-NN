from services.session_manager.service import SessionManagerService
from core.feedback_utils import FeedbackEntry
from core.model_context import ModelContext, TaskContext


def test_feedback_storage():
    service = SessionManagerService()
    sid = service.start_session()
    ctx = ModelContext(task_context=TaskContext(task_type="demo"), session_id=sid)
    service.update_context(ctx)
    fb = FeedbackEntry(
        session_id=sid,
        user_id="u1",
        agent_id="worker_demo",
        score=1,
        comment="ok",
        timestamp="t",
    )
    service.add_feedback(fb)
    stored = service.get_feedback(sid)
    assert len(stored) == 1
    assert stored[0].score == 1


from core.auto_trainer import AutoTrainer
from services.session_manager.service import SessionManagerService
from services.session_manager.feedback_store import FeedbackEntry
from core.model_context import ModelContext, TaskContext


def test_auto_trainer_adjusts_weights():
    service = SessionManagerService()
    sid = service.start_session()
    ctx = ModelContext(
        session_id=sid,
        task_context=TaskContext(task_type="docker"),
        agent_selection="worker_openhands",
        result="ok",
    )
    ctx.agents.append(
        type("Arc", (), {"agent_id": "worker_openhands", "score": 1})()
    )
    service.update_context(ctx)
    fb = FeedbackEntry(
        session_id=sid,
        user_id="u1",
        agent_id="worker_openhands",
        score=1,
        comment="good",
        timestamp="t",
    )
    service.add_feedback(fb)

    trainer = AutoTrainer(service)
    trainer.run()
    assert trainer.weights.get("worker_openhands", 0) > 0


import tempfile

from services.session_manager.service import SessionManagerService
from core.model_context import ModelContext, TaskContext


def test_memory_flow(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("SESSIONS_DIR", f"{tmp}/sessions")
        monkeypatch.setenv("MEMORY_LOG_DIR", f"{tmp}/memory")
        monkeypatch.setenv("DEFAULT_STORE_BACKEND", "file")
        monkeypatch.setenv("MEMORY_STORE_BACKEND", "file")
        service = SessionManagerService()
        sid = service.start_session()
        ctx1 = ModelContext(task_context=TaskContext(task_type="demo", description="first"), session_id=sid, result="one")
        service.update_context(ctx1)
        service2 = SessionManagerService()
        history = service2.get_context(sid)
        assert history[-1].memory[0]["output"] == "one"

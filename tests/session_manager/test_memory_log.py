import json
import tempfile
from pathlib import Path

from services.session_manager.service import SessionManagerService
from core.model_context import ModelContext, TaskContext


def test_memory_log_persistence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("SESSIONS_DIR", f"{tmp}/sessions")
        monkeypatch.setenv("MEMORY_LOG_DIR", f"{tmp}/memory")
        monkeypatch.setenv("DEFAULT_STORE_BACKEND", "file")
        monkeypatch.setenv("MEMORY_STORE_BACKEND", "file")
        service = SessionManagerService()
        sid = service.start_session()
        ctx = ModelContext(
            task_context=TaskContext(task_type="demo", description="p"),
            session_id=sid,
            result="out",
            agent_selection="a1",
        )
        service.update_context(ctx)
        log_file = Path(tmp) / "memory" / f"{sid}.json"
        assert log_file.exists()
        data = json.loads(log_file.read_text())
        assert data[0]["output"] == "out"

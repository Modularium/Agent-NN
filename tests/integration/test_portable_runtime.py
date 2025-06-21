import tempfile

from core.model_context import ModelContext, TaskContext
from services.session_manager.service import SessionManagerService
from services.vector_store.service import VectorStoreService


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout=10):
        return DummyResp({"embedding": [float(len(json["text"]))], "provider": "dummy"})


def test_portable_runtime(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("SESSIONS_DIR", f"{tmp}/sessions")
        monkeypatch.setenv("VECTOR_DB_DIR", f"{tmp}/vdb")
        monkeypatch.setenv("DEFAULT_STORE_BACKEND", "file")
        monkeypatch.setenv("VECTOR_DB_BACKEND", "chromadb")

        sm = SessionManagerService()
        sid = sm.start_session()
        ctx = ModelContext(task_context=TaskContext(task_type="demo"), session_id=sid)
        sm.update_context(ctx)
        assert sm.get_context(sid)

        vs = VectorStoreService(llm_url="http://llm")
        doc_id = vs.add_document("foo", "test")
        vs2 = VectorStoreService(llm_url="http://llm")
        res = vs2.search("foo", "test")
        assert any(m["id"] == doc_id for m in res["matches"])

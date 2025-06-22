from services.agent_worker.sample_agent.service import SampleAgentService
from core.model_context import ModelContext, TaskContext
from core.crypto import generate_keypair, verify_signature


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

    def post(self, url, json=None, timeout=10):
        if url.endswith("/generate"):
            return DummyResp(
                {"completion": "hi", "tokens_used": 1, "provider": "dummy"}
            )
        if url.endswith("/update_context") or url.endswith("/vector_search"):
            return DummyResp({})
        raise AssertionError("unexpected url" + url)


def test_worker_signs(monkeypatch, tmp_path):
    monkeypatch.setenv("KEY_DIR", str(tmp_path))
    generate_keypair("sample_agent")
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    service = SampleAgentService(llm_url="http://llm")
    ctx = ModelContext(task_context=TaskContext(task_type="demo"))
    out = service.run(ctx)
    assert out.signature
    payload = out.model_dump(exclude={"signature", "signed_by"})
    assert verify_signature(out.signed_by, payload, out.signature)

from services.task_dispatcher.service import TaskDispatcherService
from core.model_context import TaskContext, ModelContext
from core.crypto import generate_keypair, sign_payload, verify_signature
from core.governance import AgentContract


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
        if url.endswith("/run"):
            ctx = ModelContext(result="ok", metrics={"tokens_used": 1})
            sig = sign_payload("a1", ctx.model_dump(exclude={"signature", "signed_by"}))
            ctx.signed_by = sig["signed_by"]
            ctx.signature = sig["signature"]
            return DummyResp(ctx.model_dump())
        return DummyResp({})


def test_dispatcher_verifies_signature(monkeypatch, tmp_path):
    monkeypatch.setenv("KEY_DIR", str(tmp_path))
    generate_keypair("a1")
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="a1",
        allowed_roles=["demo"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
        require_signature=True,
    ).save()
    service = TaskDispatcherService()
    monkeypatch.setattr(
        service,
        "_fetch_agents",
        lambda c: [{"id": "a1", "name": "a1", "url": "http://a1"}],
    )
    monkeypatch.setattr(service, "_fetch_history", lambda s: [])
    ctx = service.dispatch_task(TaskContext(task_type="demo"))
    assert ctx.signature
    payload = ctx.model_dump(exclude={"signature", "signed_by"})
    assert verify_signature(ctx.signed_by, payload, ctx.signature)
    assert ctx.warning is None

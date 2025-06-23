import json
from sdk.client import AgentClient


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def __init__(self):
        self.sent = []

    def post(self, url, json=None, headers=None):
        self.sent.append((url, json))
        return DummyResponse({"ok": True})

    def get(self, url, headers=None):
        self.sent.append((url, None))
        return DummyResponse({"items": []})


def test_submit_task(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr("httpx.Client", lambda base_url: dummy)
    client = AgentClient()
    result = client.submit_task("demo")
    assert result["ok"] is True
    assert dummy.sent[0][0].endswith("/task")


def test_list_agents(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr("httpx.Client", lambda base_url: dummy)
    client = AgentClient()
    client.list_agents()
    assert dummy.sent[0][0].endswith("/agents")


def test_new_helpers(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr("httpx.Client", lambda base_url: dummy)
    client = AgentClient()
    client.get_agents()
    client.get_sessions()
    client.get_embeddings("hi")
    ctx = type("C", (), {"model_dump": lambda self: {"a": 1}})()
    client.dispatch_task(ctx)
    paths = [call[0] for call in dummy.sent]
    assert "/agents" in paths[0]
    assert "/sessions" in paths[1]
    assert "/embed" in paths[2]
    assert "/dispatch" in paths[3]

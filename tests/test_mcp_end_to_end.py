import json
import urllib.request

from mcp.agent_registry.service import AgentRegistryService
from mcp.session_manager.service import SessionManagerService
from mcp.task_dispatcher.service import TaskDispatcherService


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def fake_urlopen(req, timeout=0):
    url = getattr(req, "full_url", req)
    if url.endswith("/health"):
        return FakeResponse({"status": "ok"})
    if url.endswith("/execute_task"):
        data = json.loads(req.data.decode())
        return FakeResponse({"result": f"hello {data['task']}"})
    raise ValueError(url)


def test_mcp_end_to_end(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    registry = AgentRegistryService(config_path="")
    registry.register_agent(
        {
            "name": "worker_dev",
            "agent_type": "dev",
            "url": "http://worker",
            "capabilities": ["greeting"],
            "status": "online",
        }
    )

    sessions = SessionManagerService(ttl_minutes=1)
    dispatcher = TaskDispatcherService()
    dispatcher.registry = registry
    dispatcher.sessions = sessions

    sid = sessions.create_session({})
    result = dispatcher.dispatch_task("greeting", "world", sid)

    assert result["worker"] == "worker_dev"
    assert result["response"]["result"] == "hello world"
    stored = sessions.get_session(sid)
    assert stored is not None
    assert stored["history"][0]["result"]["result"] == "hello world"

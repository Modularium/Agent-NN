import json
import urllib.request

from mcp.plugin_agent_service.service import PluginAgentService


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
    return FakeResponse({"extract": "Lima"})


def test_execute_wikipedia(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    service = PluginAgentService(plugin_dir="plugins")
    result = service.execute_tool("wikipedia", {"query": "Peru"}, {})
    assert "summary" in result or "error" in result


def test_execute_filesystem(tmp_path):
    service = PluginAgentService(plugin_dir="plugins")
    file_path = tmp_path / "t.txt"
    result = service.execute_tool(
        "filesystem",
        {"action": "write", "path": str(file_path), "content": "data"},
        {},
    )
    assert result["status"] == "written"
    read_res = service.execute_tool(
        "filesystem", {"action": "read", "path": str(file_path)}, {}
    )
    assert read_res["content"] == "data"


def test_execute_unknown_tool():
    service = PluginAgentService(plugin_dir="plugins")
    result = service.execute_tool("missing_tool", {}, {})
    assert result == {"error": "unknown tool"}


def test_wikipedia_missing_query():
    service = PluginAgentService(plugin_dir="plugins")
    result = service.execute_tool("wikipedia", {}, {})
    assert result == {"error": "no query"}


def test_filesystem_missing_path():
    service = PluginAgentService(plugin_dir="plugins")
    result = service.execute_tool("filesystem", {}, {})
    assert result == {"error": "no path"}

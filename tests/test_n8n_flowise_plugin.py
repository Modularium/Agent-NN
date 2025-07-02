import httpx
from plugins.n8n_workflow.plugin import Plugin as N8NPlugin
from plugins.flowise_workflow.plugin import Plugin as FlowisePlugin


def _fake_request(method: str, url: str, json=None, headers=None, timeout=10):
    return httpx.Response(200, json={"ok": True, "url": url, "method": method, "timeout": timeout})


def test_n8n_plugin(monkeypatch):
    monkeypatch.setattr(httpx, "request", _fake_request)
    plugin = N8NPlugin()
    result = plugin.execute({"endpoint": "http://n8n.local", "path": "/webhook", "method": "POST", "timeout": 5}, {})
    assert result["status"] == "success"
    assert result["data"] == {"ok": True, "url": "http://n8n.local/webhook", "method": "POST", "timeout": 5}


def test_flowise_plugin(monkeypatch):
    monkeypatch.setattr(httpx, "request", _fake_request)
    plugin = FlowisePlugin()
    result = plugin.execute({"endpoint": "http://flowise.local", "path": "/api", "method": "POST"}, {})
    assert result["status"] == "success"
    assert result["data"] == {"ok": True, "url": "http://flowise.local/api", "method": "POST", "timeout": 10}

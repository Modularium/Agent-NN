import httpx
from plugins.n8n_workflow.plugin import Plugin as N8NPlugin
from plugins.flowise_workflow.plugin import Plugin as FlowisePlugin


def _fake_post(url: str, json=None, headers=None, timeout=10):
    return httpx.Response(200, json={"ok": True, "url": url})


def test_n8n_plugin(monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_post)
    plugin = N8NPlugin()
    result = plugin.execute({"url": "http://n8n.local/webhook"}, {})
    assert result["status"] == "success"
    assert result["data"] == {"ok": True, "url": "http://n8n.local/webhook"}


def test_flowise_plugin(monkeypatch):
    monkeypatch.setattr(httpx, "post", _fake_post)
    plugin = FlowisePlugin()
    result = plugin.execute({"url": "http://flowise.local/api"}, {})
    assert result["status"] == "success"
    assert result["data"] == {"ok": True, "url": "http://flowise.local/api"}

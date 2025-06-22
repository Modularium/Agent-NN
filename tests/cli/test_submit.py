from typer.testing import CliRunner
from sdk.cli.main import app


class DummyResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def post(self, url, json=None, headers=None):
        return DummyResp({"ok": True})

    def get(self, url, headers=None):
        return DummyResp({"items": []})


def test_cli_submit(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda base_url: DummyClient())
    runner = CliRunner()
    result = runner.invoke(app, ["submit", "test"])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_cli_ask(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda base_url: DummyClient())
    runner = CliRunner()
    result = runner.invoke(app, ["ask", "hi", "--task-type=dev"])
    assert result.exit_code == 0
    assert "ok" in result.stdout

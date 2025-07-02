import httpx

from plugins import ToolPlugin


class Plugin(ToolPlugin):
    """Trigger an n8n workflow via HTTP POST."""

    def execute(self, input: dict, context: dict) -> dict:
        url = input.get("url")
        payload = input.get("payload", {})
        headers = input.get("headers", {})
        if not url:
            return {"error": "no url provided"}
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            return {"status": "success", "data": data}
        except Exception as exc:  # pragma: no cover - network errors
            return {"error": str(exc)}

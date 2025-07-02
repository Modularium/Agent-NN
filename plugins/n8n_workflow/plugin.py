import httpx

from plugins import ToolPlugin


class Plugin(ToolPlugin):
    """Trigger an n8n workflow via HTTP request."""

    def execute(self, input: dict, context: dict) -> dict:
        url = input.get("url")
        endpoint = input.get("endpoint")
        path = input.get("path", "")
        payload = input.get("payload", {})
        headers = input.get("headers", {})
        method = input.get("method", "POST").upper()
        timeout = input.get("timeout", 10)
        auth = input.get("auth")
        if not url and not endpoint:
            return {"error": "no url or endpoint provided"}
        if not url:
            url = f"{endpoint.rstrip('/')}{path}"
        try:
            resp = httpx.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
                auth=(auth["username"], auth["password"]) if auth else None,
            )
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            return {"status": "success", "data": data}
        except Exception as exc:  # pragma: no cover - network errors
            return {"error": str(exc)}

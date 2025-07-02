import httpx

from plugins import ToolPlugin


class Plugin(ToolPlugin):
    """Call a Flowise chatflow or API endpoint."""

    def execute(self, input: dict, context: dict) -> dict:
        url = input.get("url")
        endpoint = input.get("endpoint")
        path = input.get("path", "")
        payload = input.get("payload", {})
        headers = input.get("headers", {})
        method = input.get("method", "POST").upper()
        timeout = input.get("timeout", 10)
        if not url and not endpoint:
            return {"error": "no url or endpoint provided"}
        if not url and endpoint:
            url = f"{endpoint.rstrip('/')}{path}"
        elif not url:
            return {"error": "no url or endpoint provided"}
        try:
            resp = httpx.request(method, url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            return {"status": "success", "data": data}
        except Exception as exc:  # pragma: no cover - network errors
            return {"error": str(exc)}

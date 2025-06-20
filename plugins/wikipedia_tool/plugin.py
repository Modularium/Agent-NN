import json
import urllib.parse
import urllib.request

from plugins import ToolPlugin


class Plugin(ToolPlugin):
    """Retrieve a short summary from Wikipedia."""

    def execute(self, input: dict, context: dict) -> dict:
        term = input.get("query")
        if not term:
            return {"error": "no query"}
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
            term
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return {"summary": data.get("extract", "")}
        except Exception as exc:  # pragma: no cover - network errors
            return {"error": str(exc)}

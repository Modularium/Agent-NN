import os

from plugins import ToolPlugin


class Plugin(ToolPlugin):
    """Simple file read/write utility."""

    def execute(self, input: dict, context: dict) -> dict:
        action = input.get("action", "read")
        path = input.get("path")
        if not path:
            return {"error": "no path"}
        if action == "write":
            content = input.get("content", "")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return {"status": "written", "path": path}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            return {"content": content}
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}

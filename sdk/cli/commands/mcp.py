from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer

from core.run_service import run_service
from agentnn.mcp.mcp_server import create_app

mcp_app = typer.Typer(name="mcp", help="MCP utilities")

_CONFIG = Path.home() / ".agentnn" / "mcp_endpoints.json"


def _load() -> dict[str, str]:
    if _CONFIG.exists():
        try:
            return json.loads(_CONFIG.read_text())
        except Exception:
            return {}
    return {}


def _save(data: dict[str, str]) -> None:
    _CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG.write_text(json.dumps(data, indent=2))


@mcp_app.command("serve")
def serve(host: str = "0.0.0.0", port: int = 8090) -> None:
    """Start the MCP server."""
    app = create_app()
    run_service(app, host=host, port=port)


@mcp_app.command("register-endpoint")
def register_endpoint(alias: str, url: str) -> None:
    """Store alias and URL for a remote MCP server."""
    data = _load()
    data[alias] = url.rstrip("/")
    _save(data)
    typer.echo(f"registered {alias}")


@mcp_app.command("invoke")
def invoke(target: str, input: str = typer.Option("{}", "--input")) -> None:
    """Invoke TOOL_ID on endpoint alias with JSON input."""
    if "." not in target:
        typer.echo("invalid target")
        raise typer.Exit(1)
    alias, tool_id = target.split(".", 1)
    data = _load()
    if alias not in data:
        typer.echo("unknown endpoint")
        raise typer.Exit(1)
    url = f"{data[alias]}/v1/mcp/{tool_id}"
    payload = json.loads(input)
    resp = httpx.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    typer.echo(resp.text)


__all__ = ["mcp_app"]

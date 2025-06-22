"""Minimal CLI to interact with MCP services."""
import json
import urllib.request
import typer

DISPATCHER_URL = "http://localhost:8000"

app = typer.Typer()


@app.command()
def dispatch(task: str):
    """Send ``task`` to the Task Dispatcher and output the result."""
    req = urllib.request.Request(
        f"{DISPATCHER_URL}/dispatch",
        data=json.dumps({"task": task}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode())


if __name__ == "__main__":
    app()

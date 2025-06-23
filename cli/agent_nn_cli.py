import json
import os
import click
import requests

API_URL = os.getenv("AGENT_NN_API", "http://localhost:8000")


def post(path: str, data: dict) -> dict:
    resp = requests.post(f"{API_URL}{path}", json=data)
    resp.raise_for_status()
    return resp.json()


def get(path: str) -> dict:
    resp = requests.get(f"{API_URL}{path}")
    resp.raise_for_status()
    return resp.json()


@click.group()
def cli() -> None:
    """Simple Agent-NN CLI."""


@cli.command()
@click.option("--agent", default="dev", help="Agent name")
@click.option("--task-type", default="chat", help="Task type")
@click.option("--session-id", help="Session id")
@click.option("--input", "message", help="Question text")
@click.option("--interactive", is_flag=True, help="Interactive mode")
def ask(agent: str, task_type: str, session_id: str | None, message: str | None, interactive: bool) -> None:
    """Send a message to an agent."""
    sid = session_id
    def send(msg: str) -> None:
        nonlocal sid
        payload = {"task_type": task_type, "input": msg, "agent": agent}
        if sid:
            payload["session_id"] = sid
        result = post("/task", payload)
        sid = result.get("session_id", sid)
        resp = result.get("result")
        if isinstance(resp, dict) and resp.get("operation_id"):
            click.echo(f"{resp.get('status')} - ID {resp.get('operation_id')}")
        else:
            click.echo(resp)
    if interactive:
        while True:
            msg = click.prompt(">>") if message is None else message
            if msg.lower() in {"exit", "quit"}:
                break
            send(msg)
            message = None
    else:
        if not message:
            raise click.UsageError("--input required without --interactive")
        send(message)
        if not session_id:
            click.echo(f"Session: {sid}")


@cli.command()
def tools() -> None:
    """List available agents."""
    data = get("/agents")
    click.echo(json.dumps(data, indent=2))


@cli.command()
def sessions() -> None:
    """List active sessions."""
    data = get("/sessions")
    click.echo(json.dumps(data, indent=2))


@cli.command()
@click.argument("session_id")
def log(session_id: str) -> None:
    """Show chat history for SESSION_ID."""
    data = get(f"/chat/history/{session_id}")
    click.echo(json.dumps(data, indent=2))


@cli.command()
@click.argument("session_id")
@click.argument("index", type=int)
@click.option("--rating", type=click.Choice(["good", "bad"]), required=True)
@click.option("--comment", default="")
def feedback(session_id: str, index: int, rating: str, comment: str) -> None:
    """Send feedback for a chat message."""
    data = post("/chat/feedback", {"session_id": session_id, "index": index, "rating": rating, "comment": comment})
    click.echo(json.dumps(data))


if __name__ == "__main__":
    cli()

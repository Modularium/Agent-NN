from __future__ import annotations

from pathlib import Path
import typer
import yaml
import re

from ..utils.history import log_entry

from .session import start as session_start
from ..utils.io import ensure_parent


quickstart_app = typer.Typer(name="quickstart", help="Automated setup helpers")


def _gen_from_desc(desc: str) -> dict:
    slug = re.sub(r"\W+", "-", desc.lower()).strip("-")[:15] or "agent"
    role = "planner" if "plan" in desc.lower() else "assistant"
    return {"id": slug, "role": role, "description": desc, "tools": []}


@quickstart_app.command("agent")
def quickstart_agent(
    name: str | None = typer.Option(None),
    role: str | None = typer.Option(None),
    output: Path = typer.Option(Path("agent.yaml")),
    from_description: str | None = typer.Option(None, "--from-description"),
) -> None:
    """Create a new agent config from template or description."""
    if from_description:
        data = _gen_from_desc(from_description)
    else:
        built_in = (
            Path(__file__).resolve().parent.parent / "templates" / "agent_template.yaml"
        )
        data = yaml.safe_load(built_in.read_text())
    if name:
        data["id"] = name
    if role:
        data["role"] = role
    ensure_parent(output)
    output.write_text(yaml.safe_dump(data))
    log_entry("quickstart_agent", {"output": str(output)})
    typer.echo(str(output))


@quickstart_app.command("session")
def quickstart_session(
    template: str | None = None,
    from_: Path | None = typer.Option(None, "--from"),
    complete: bool = typer.Option(False, "--complete"),
) -> None:
    """Start a session using a template."""
    from ..config import CLIConfig

    cfg = CLIConfig.load()
    if from_:
        data = yaml.safe_load(from_.read_text())
        if complete:
            built_in = (
                Path(__file__).resolve().parent.parent / "templates" / "session_template.yaml"
            )
            defaults = yaml.safe_load(built_in.read_text())
            for k, v in defaults.items():
                data.setdefault(k, v)
            tmp = from_.with_suffix(".complete.yaml")
            tmp.write_text(yaml.safe_dump(data))
            path = tmp
        else:
            path = from_
    else:
        path = Path(template or cfg.default_session_template)
    log_entry("quickstart_session", {"template": str(path)})
    session_start(template=path)


__all__ = ["quickstart_app"]

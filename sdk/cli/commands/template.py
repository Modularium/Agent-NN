from __future__ import annotations

import json
import os
from pathlib import Path
import typer
import yaml
from rich.console import Console
from rich.syntax import Syntax
import jsonschema

from ..utils.io import ensure_parent


template_app = typer.Typer(name="template", help="Manage templates")


@template_app.command("list")
def template_list() -> None:
    """List available templates."""
    from ..config import CLIConfig

    cfg = CLIConfig.load()
    directory = Path(os.path.expanduser(cfg.templates_dir))
    if not directory.exists():
        typer.echo("no templates")
        return
    for item in directory.iterdir():
        if item.is_file():
            typer.echo(item.name)


@template_app.command("show")
def template_show(name: str, as_: str = typer.Option("yaml", "--as")) -> None:
    """Show template contents."""
    from ..config import CLIConfig

    cfg = CLIConfig.load()
    path = Path(os.path.expanduser(cfg.templates_dir)) / name
    if not path.exists():
        typer.echo("template not found")
        raise typer.Exit(code=1)
    text = path.read_text()
    if as_ == "json":
        data = yaml.safe_load(text)
        typer.echo(json.dumps(data, indent=2))
    else:
        console = Console()
        console.print(Syntax(text, "yaml", line_numbers=True))


@template_app.command("doc")
def template_doc(name: str) -> None:
    """Output Markdown documentation for template."""
    from ..config import CLIConfig

    cfg = CLIConfig.load()
    path = Path(os.path.expanduser(cfg.templates_dir)) / name
    if not path.exists():
        typer.echo("template not found")
        raise typer.Exit(code=1)
    text = path.read_text()
    typer.echo("```yaml\n" + text + "\n```")


@template_app.command("validate")
def template_validate(path: Path, kind: str | None = None) -> None:
    """Validate template against built-in schema."""
    text = path.read_text()
    data = yaml.safe_load(text)
    if not kind:
        if isinstance(data, dict) and "agents" in data and "tasks" in data:
            kind = "session"
        elif isinstance(data, dict) and "task" in data:
            kind = "task"
        else:
            kind = "agent"
    schema_file = (
        Path(__file__).resolve().parent.parent / "schemas" / f"{kind}_schema.json"
    )
    schema = json.loads(schema_file.read_text())
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        typer.secho("Ungültig", fg=typer.colors.RED)
        typer.echo(str(exc).split("\n", 1)[0])
        raise typer.Exit(1)
    typer.secho("Gültig", fg=typer.colors.GREEN)


@template_app.command("init")
def template_init(kind: str, output: Path) -> None:
    """Create template file from built-in defaults."""
    built_in = (
        Path(__file__).resolve().parent.parent / "templates" / f"{kind}_template.yaml"
    )
    if not built_in.exists():
        typer.echo("unknown template kind")
        raise typer.Exit(code=1)
    ensure_parent(output)
    output.write_text(built_in.read_text())
    typer.echo(str(output))


__all__ = ["template_app", "template_validate", "template_doc", "template_show"]

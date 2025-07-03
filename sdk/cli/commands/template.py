from __future__ import annotations

import os
from pathlib import Path
import typer

from ..config import CLIConfig
from ..utils.io import ensure_parent


template_app = typer.Typer(name="template", help="Manage templates")


@template_app.command("list")
def template_list() -> None:
    """List available templates."""
    cfg = CLIConfig.load()
    directory = Path(os.path.expanduser(cfg.templates_dir))
    if not directory.exists():
        typer.echo("no templates")
        return
    for item in directory.iterdir():
        if item.is_file():
            typer.echo(item.name)


@template_app.command("show")
def template_show(name: str) -> None:
    """Show template contents."""
    cfg = CLIConfig.load()
    path = Path(os.path.expanduser(cfg.templates_dir)) / name
    if not path.exists():
        typer.echo("template not found")
        raise typer.Exit(code=1)
    typer.echo(path.read_text())


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


__all__ = ["template_app"]

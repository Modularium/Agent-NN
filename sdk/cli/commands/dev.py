from __future__ import annotations

from pathlib import Path

import typer

from tools.cli_docgen import update_cli_doc


dev_app = typer.Typer(name="dev", help="Developer utilities")


@dev_app.command("docgen")
def docgen(output: Path = typer.Option(Path("docs/cli.md"), "--output")) -> None:
    """Generate CLI reference documentation."""
    update_cli_doc(output)
    typer.echo(str(output))


__all__ = ["dev_app"]

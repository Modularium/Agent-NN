from __future__ import annotations

import typer

from agentnn.prompting import propose_refinement, evaluate_prompt_quality

prompt_app = typer.Typer(name="prompt", help="Prompt utilities")


@prompt_app.command("refine")
def refine_prompt(input: str, strategy: str = typer.Option("direct", "--strategy")) -> None:
    """Return a refined prompt string."""
    typer.echo(propose_refinement(input, strategy))


@prompt_app.command("quality")
def quality_prompt(input: str) -> None:
    """Evaluate quality of INPUT and output score."""
    score = evaluate_prompt_quality(input)
    typer.echo(str(score))


__all__ = ["prompt_app"]

import typer
import httpx


def handle_http_error(err: httpx.HTTPStatusError) -> None:
    """Show a friendly message for HTTP errors."""
    if err.response.status_code == 401:
        typer.secho(
            "\u26d4 Nicht autorisiert – überprüfe deinen API-Key",
            fg=typer.colors.RED,
        )
    else:
        typer.secho(f"HTTP Error: {err.response.status_code}", fg=typer.colors.RED)
    raise typer.Exit(1)

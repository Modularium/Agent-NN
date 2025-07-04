# Developer Guide: Extending the CLI

Dieses Dokument erklärt, wie neue Unterbefehle für `agentnn` erstellt werden.

## Neues Subkommando anlegen

1. Erzeuge ein Modul unter `sdk/cli/commands/`:

   ```python
   # sdk/cli/commands/example.py
   import typer

   example_app = typer.Typer(name="example", help="Beispielbefehle")

   @example_app.command("hello")
   def hello(name: str = "World") -> None:
       """Gibt eine Begrüßung aus."""
       typer.echo(f"Hello {name}")
   ```

2. Registriere das neue Typer-Objekt in `sdk/cli/main.py`:

   ```python
   from .commands.example import example_app
   app.add_typer(example_app, name="example")
   ```

3. Füge falls nötig Tests unter `tests/cli/` hinzu und dokumentiere das Kommando.

## Konventionen

- Alle Befehle sollten englische Funktionsnamen besitzen und eine kurze Hilfe über `help=` angeben.
- Längere Erläuterungen oder Beispiele kommen in `epilog=`.
- Für optionale Dokumentationsausgabe kann das Hilfs-Callback `doc_printer()` verwendet werden.
- Logging und Fehlerausgaben erfolgen über die Funktionen in `sdk/cli/utils/formatting.py`.

## Beispielaufruf

```bash
agentnn example hello --name Alice
```

Weitere Beispiele befinden sich in `docs/cli.md`.

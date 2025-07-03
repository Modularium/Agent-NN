# AgentNN CLI

`agentnn` is the unified command line interface for all services of Agent‑NN.
Install the project and run `agentnn --help` to see available commands.

## Subcommands

| Command | Description |
|---------|-------------|
| `session` | manage and track conversation sessions |
| `context` | export stored context data and context maps |
| `agent` | inspect and update agent profiles |
| `task` | queue and inspect tasks |
| `model` | list and switch language models |
| `prompt` | refine prompts and check quality |
| `config` | show effective configuration |
| `governance` | governance and trust utilities |

## Examples

```bash
agentnn session start examples/demo.yaml
agentnn context export mysession --out demo_context.json
agentnn agent register config/agent.yaml
```

## Global Flags

- `--version` – show version and exit
- `--token` – override API token for this call
- `--help` – display help for any command
- `--verbose` – detailed log output
- `--quiet` – suppress info messages
- `--debug` – show stack traces on errors

## \U0001F4C0 Ausgabeformate & interaktive Nutzung

Viele `list`-Befehle unterstützen das Flag `--output` mit den Optionen
`table`, `json` oder `markdown`.

```bash
agentnn agent list --output markdown
```

Der Befehl `agent register --interactive` startet einen kurzen Wizard und
fragt Name, Rolle, Tools und Beschreibung interaktiv ab.

Session templates are YAML files containing `agents` and `tasks` sections.
The CLI prints JSON output so that results can easily be processed in scripts.
Check file paths and YAML formatting if a command reports errors.

## Alte CLI ersetzt

Vor der Modularisierung gab es mehrere Einstiegspunkte wie `cli/agentctl.py`
oder `mcp/cli.py`. Alle Funktionen wurden in `agentnn` konsolidiert. Beispiele
zur Migration:

```bash
python cli/agentctl.py deploy config/agent.yaml  # alt
agentnn agent register config/agent.yaml         # neu
```

## 🧩 CLI-Architektur & Interna

Die Befehle der CLI sind modular aufgebaut. Jedes Subkommando lebt in
`sdk/cli/commands/` als eigenständiges Modul. Hilfsfunktionen sind im
Verzeichnis `sdk/cli/utils/` gekapselt und in `formatting.py` bzw. `io.py`
strukturiert. Dadurch können neue Kommandos leicht angelegt werden, ohne
unerwünschte Abhängigkeiten zu erzeugen. `main.py` bindet lediglich die
bereits initialisierten `Typer`-Instanzen ein und enthält keine Logik
oder Rückimporte.

# AgentNN CLI

`agentnn` is the unified command line interface for all services of Agent‑NN.
Install the project and run `agentnn --help` to see available commands.

## Subcommands

| Command | Description |
|---------|-------------|
| `session` | manage and track conversation sessions |
| `context` | export stored context data |
| `agent` | inspect and update agent profiles |
| `task` | queue and inspect tasks |
| `model` | list and switch language models |
| `config` | show effective configuration |
| `governance` | governance and trust utilities |

## Examples

```bash
agentnn session start examples/demo.yaml
agentnn context export mysession --out demo_context.json
agentnn agent deploy --config config/agent.yaml
```

## Global Flags

- `--version` – show version and exit
- `--token` – override API token for this call
- `--help` – display help for any command

Session templates are YAML files containing `agents` and `tasks` sections.
The CLI prints JSON output so that results can easily be processed in scripts.
Check file paths and YAML formatting if a command reports errors.

## 🧩 CLI-Architektur & Interna

Die Befehle der CLI sind modular aufgebaut. Jedes Subkommando lebt in
`sdk/cli/commands/` als eigenständiges Modul. Hilfsfunktionen sind im
Verzeichnis `sdk/cli/utils/` gekapselt und in `formatting.py` bzw. `io.py`
strukturiert. Dadurch können neue Kommandos leicht angelegt werden, ohne
unerwünschte Abhängigkeiten zu erzeugen. `main.py` bindet lediglich die
bereits initialisierten `Typer`-Instanzen ein und enthält keine Logik
oder Rückimporte.

# AgentNN CLI

`agentnn` is the unified command line interface for all services of Agent‑NN.
Install the project and run `agentnn --help` to see available commands.

## Subcommands

| Command | Description | Source |
|---------|-------------|--------|
| `agent` | Agent management | [agent.py](../sdk/cli/commands/agent.py) |
| `context` | Context utilities | [context.py](../sdk/cli/commands/context.py) |
| `dev` | Developer utilities | [dev.py](../sdk/cli/commands/dev.py) |
| `feedback` | Feedback utilities | [feedback.py](../sdk/cli/commands/feedback.py) |
| `prompt` | Prompt utilities | [prompt.py](../sdk/cli/commands/prompt.py) |
| `quickstart` | Automated setup helpers | [quickstart.py](../sdk/cli/commands/quickstart.py) |
| `session` | Session utilities | [session.py](../sdk/cli/commands/session.py) |
| `template` | Manage templates | [template.py](../sdk/cli/commands/template.py) |
| `tools` | Tool registry | [tools.py](../sdk/cli/commands/tools.py) |
| `train` | Training management | [train.py](../sdk/cli/commands/train.py) |
| `ask` | Send a quick task to the dispatcher. | [root.py](../sdk/cli/commands/root.py) |
| `promote` | Promote a queued task by id. | [root.py](../sdk/cli/commands/root.py) |
| `rate` | Submit a peer rating. | [root.py](../sdk/cli/commands/root.py) |
| `reset` | Delete session history and user configuration. | [root.py](../sdk/cli/commands/root.py) |
| `sessions` | List active sessions. | [root.py](../sdk/cli/commands/root.py) |
| `submit` | Submit a task to the dispatcher. | [root.py](../sdk/cli/commands/root.py) |
| `verify` | Verify the signature of a ModelContext JSON file. | [root.py](../sdk/cli/commands/root.py) |

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

## Ausgabeformate & interaktive Nutzung

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

## 🧩 CLI-Architektur & Interna

Die Befehle der CLI sind modular aufgebaut. Jedes Subkommando lebt in
`sdk/cli/commands/` als eigenständiges Modul. Hilfsfunktionen sind im
Verzeichnis `sdk/cli/utils/` gekapselt und in `formatting.py` bzw. `io.py`
strukturiert. Dadurch können neue Kommandos leicht angelegt werden, ohne
unerwünschte Abhängigkeiten zu erzeugen. `main.py` bindet lediglich die
bereits initialisierten `Typer`-Instanzen ein und enthält keine Logik
oder Rückimporte.

## ⚙️ Konfiguration & Vorlagen

Beim Start sucht die CLI nach `~/.agentnn/config.toml` und liest globale
Standardwerte wie `default_session_template`, `output_format` und `log_level`.
Eine optionale `agentnn.toml` im aktuellen Projektverzeichnis kann diese
Einstellungen überschreiben.

Beispiel `agentnn.toml`:

```toml
output_format = "json"
default_session_template = "project/session.yaml"
```

Vorlagen liegen unter `~/.agentnn/templates/` und können mit folgenden
Befehlen verwaltet werden:

```bash
agentnn template list
agentnn template show session_template.yaml
agentnn template init session --output=my.yaml
```

Quickstart-Kürzel kombinieren Konfiguration und Vorlagen:

```bash
agentnn quickstart agent --name Demo --role planner
agentnn quickstart session --template demo_session.yaml
```

## Resetting local data

Mit `agentnn reset --confirm` werden die lokale Kontextdatenbank,
gespeicherte Snapshots und das Verzeichnis `~/.agentnn` entfernt.
Nutze diesen Befehl, um deine Umgebung komplett zurückzusetzen.

## Intelligente Templates & Automatisierung

Die Template-Befehle unterstützen nun Schema-Validierung und Konvertierung. Beispiele:

```bash
agentnn template validate my_agent.yaml
agentnn template show my_agent.yaml --as json
agentnn template doc my_agent.yaml > AGENT.md
```

Mit `quickstart agent --from-description` lässt sich aus einer Kurzbeschreibung automatisch ein Agent-Template erzeugen:

```bash
agentnn quickstart agent --from-description "Planender Entscheidungsagent mit Zugriff auf Tools" --output agent-smart.yaml
```

Unvollständige Session-Templates können über `quickstart session --from=partial.yaml --complete` ergänzt werden. Alle Aufrufe werden im Verzeichnis `~/.agentnn/history/` protokolliert.

## 🧙 Interaktive Wizards – Schritt für Schritt

Viele Befehle besitzen einen geführten Wizard-Modus. Er wird mit `--interactive`
oder über `agentnn quickstart` aktiviert und fragt alle notwendigen Angaben
nacheinander ab.

### Agent registrieren

```bash
agentnn agent register --interactive
```

Beispielausgabe:

```
Agent name: DemoAgent
Role [assistant]: planner
Tools (comma separated): search
Description: Beispielagent
```

Nach Bestätigung wird die Konfiguration an die Registry gesendet und unter
`~/.agentnn/history/` protokolliert.

### Session starten

```bash
agentnn quickstart session
```

Der Wizard erstellt eine Session-Vorlage aus den Standardwerten und startet
eine neue Unterhaltung. Mit `--from=<datei>` kann eine vorhandene Vorlage
verwendet werden. Das Flag `--complete` ergänzt fehlende Felder automatisch.

### Weitere Tipps

- `--preset` lädt gespeicherte Einstellungen aus `~/.agentnn/presets/`.
- `--last` nutzt die zuletzt verwendete Vorlage erneut.
- Abgebrochene Wizards lassen sich jederzeit neu starten.

# Entwickler:innen-Setup

Dieses Projekt verwendet [Poetry](https://python-poetry.org/) für das Dependency-Management und Packaging.

## Lokale Installation

```bash
poetry install
```

Falls das Packaging fehlschlägt, kann die Installation ohne Projekt-Code mit `--no-root` erfolgen:

```bash
poetry install --no-root
```

Alternativ lässt sich der Paketmodus dauerhaft in der `pyproject.toml` deaktivieren:

```toml
[tool.poetry]
package-mode = false
```

Nach der Installation steht der CLI-Befehl `agentnn` zur Verfügung:

```bash
poetry run agentnn --help
```

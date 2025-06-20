# Entwicklungssetup

Dieses Dokument beschreibt, wie Sie eine lokale Entwicklungsumgebung für Agent-NN einrichten.

## Voraussetzungen

- Python >= 3.10
- Git
- Docker (für die Services)

## Repository klonen

```bash
git clone https://github.com/EcoSphereNetwork/Smolit_LLM-NN.git
cd Smolit_LLM-NN
```

## Abhängigkeiten installieren

```bash
pip install -r requirements.txt
pip install -r test-requirements.txt  # optional für Tests
```

## Services starten

Die wichtigsten MCP-Services können über Docker Compose gestartet werden:

```bash
docker-compose up dispatcher registry session-manager
```

Nun ist der Dispatcher unter `http://localhost:8000` erreichbar.

## Tests ausführen

```bash
./tests/ci_check.sh
```

## Linting

```bash
ruff check .
mypy mcp
```

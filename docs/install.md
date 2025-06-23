# Installation

## From Source

```bash
poetry install --with sdk
poetry run agentnn --version
```

## Using pip

```bash
pip install agentnn
```

## Docker

Alle Services lassen sich mittels Docker Compose starten:

```bash
docker-compose up dispatcher registry session-manager vector-store llm-gateway
```

### Empfohlene Umgebung

- Python 3.9 oder neuer
- Mindestens 4 GB RAM (8 GB empfohlen)

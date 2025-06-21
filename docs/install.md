# Installation

## From Source

```bash
pip install -r requirements.txt
pip install -e .[sdk]
agentnn --version
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

# Installation

## From Source

```bash
pip install -r requirements.txt
pip install -e .
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

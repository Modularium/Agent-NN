# Release v1.0.0-mcp

Dies ist die erste stabile Ausgabe der Modular Control Plane fuer Agent-NN.

## Highlights
- Alle Kernservices als eigenstaendige Container
- Docker-Compose fuer Entwicklung und Produktion
- Python SDK aus den OpenAPI-Spezifikationen generiert

## Systemanforderungen
- Docker 20+
- Python 3.10 fuer SDK und Tools

## Installation
```bash
docker compose -f docker-compose.production.yml up -d
```

## Bekannte Risiken & TODOs
- Authentifizierung ist noch rudimentaer
- Weitere Worker-Services koennen folgen

# Setup Guide

Dieser Leitfaden beschreibt die Installation von Agent-NN Schritt für Schritt.

## Vorbereitung

* Node.js 18+
* Docker mit Docker Compose
* Python 3.10+
* [Poetry](https://python-poetry.org/)

Kopiere die Datei `.env.example` zu `.env` und passe Werte wie Ports oder Tokens an.

## Python / Poetry

```bash
poetry install
```

Die CLI steht anschließend über `poetry run agentnn` bereit.

## Frontend bauen

```bash
./scripts/deploy/build_frontend.sh
```

Die statischen Dateien landen in `frontend/dist/`.

## Dienste starten

```bash
./scripts/deploy/start_services.sh --build
```

Die Container laufen im Hintergrund. Beende sie mit `docker compose down`.

### Docker Compose vs. lokal

Alle Services lassen sich auch direkt mit `docker compose up` starten. Für eine rein lokale Ausführung müssen Redis und Postgres installiert sein.

## Umgebungsvariablen

Alle benötigten Variablen sind in `.env.example` dokumentiert. Kopiere diese Datei nach `.env` und passe sie an deine Umgebung an.

## FAQ

**Fehler `unknown flag: -d`**
: Stelle sicher, dass `docker compose` statt `docker` verwendet wird und deine Docker-Version aktuell ist.

**Ports bereits belegt**
: Passe die Ports in `.env` an oder stoppe den blockierenden Dienst.


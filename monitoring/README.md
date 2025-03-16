# Agent-NN Dashboard

Ein modernes Dashboard für die Überwachung und Verwaltung des Agent-NN Systems.

## Funktionen

- Systemübersicht
- Agent-Verwaltung
- Modell-Verwaltung
- Knowledge-Base-Verwaltung
- Monitoring
- Sicherheit
- A/B-Tests
- Einstellungen
- Logs und Alerts
- Dokumentation

## Installation

### Mit Docker

```bash
docker-compose up -d
```

### Manuelle Installation

Backend:
```bash
cd monitoring/api
pip install -r requirements.txt
uvicorn server:app --reload
```

Frontend:
```bash
cd monitoring/dashboard
npm install
npm start
```

## Zugriff

Dashboard: http://localhost:3000
API: http://localhost:8000

Demo-Zugangsdaten:
- Benutzername: admin
- Passwort: password

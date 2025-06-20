# Fehlercodes der MCP-API

| Code | Beschreibung                          |
|-----|--------------------------------------|
| 400 | Ungültige Anfrage oder Parameter      |
| 404 | Ressource nicht gefunden             |
| 500 | Interner Fehler im Service           |
| 503 | Abhängiger Dienst nicht erreichbar   |

Alle Endpunkte liefern ein JSON-Objekt der Form `{"error": "message"}` im Fehlerfall.

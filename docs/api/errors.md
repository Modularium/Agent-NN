# Fehlercodes der MCP-API

| Code | Beschreibung                          |
|-----|--------------------------------------|
| 400 | Ungültige Anfrage oder Parameter      |
| 404 | Ressource nicht gefunden             |
| 500 | Interner Fehler im Service           |
| 503 | Abhängiger Dienst nicht erreichbar   |

Alle Endpunkte liefern ein JSON-Objekt mit folgenden Feldern:

```json
{
  "status": "error",
  "detail": "Fehlerbeschreibung",
  "code": 400
}
```

`code` ist optional und entspricht dem HTTP-Status, falls angegeben.

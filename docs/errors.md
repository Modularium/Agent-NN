# Fehlerobjekte

Alle Services geben Fehler im folgenden Format zurueck:

```json
{
  "status": "error",
  "detail": "Beschreibung des Fehlers",
  "code": 400
}
```

Die Felder `status` und `detail` sind immer vorhanden. `code` ist optional und
orientiert sich an dem HTTP-Status-Code des Fehlers.

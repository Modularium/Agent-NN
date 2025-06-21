# Anwendungsbeispiele

## 1. Textgenerierung

```bash
agentnn submit --prompt "Write a poem about stars"
```

Erwartete Ausgabe (gekürzt):

```json
{
  "provider": "openai",
  "tokens": 42,
  "result": "..."
}
```

## 2. Semantische Suche

Listen verfügbare Modelle und führe dann eine Anfrage aus:

```bash
agentnn model list
agentnn submit --task semantic --prompt "Suche Artikel zu Energie"
```

Die Antwort enthält passende Dokumente inklusive Distanzwerten.

## 3. Dialog-Sitzung

Starte eine Sitzung und schicke Folgefragen:

```bash
agentnn session start
agentnn submit --session <id> --prompt "Wie funktioniert das Routing?"
agentnn submit --session <id> --prompt "Gibt es ein Beispiel?"
```

Der Kontext bleibt erhalten, so dass der zweite Aufruf auf der vorherigen Frage aufbaut.

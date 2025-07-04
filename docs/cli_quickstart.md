# CLI Quickstart

Diese Anleitung führt in wenigen Schritten von der Installation bis zur ersten ausgeführten Aufgabe.

1. **Agent anlegen**
   
   ```bash
   agentnn quickstart agent --name MyAgent
   ```
   
   Dadurch entsteht eine einfache `agent.yaml` im aktuellen Verzeichnis.

2. **Session starten**
   
   ```bash
   agentnn session start agent.yaml
   ```
   
   Die CLI gibt eine `session_id` aus, unter der alle weiteren Aufgaben laufen.

3. **Aufgabe übermitteln**
   
   ```bash
   echo '{"task": "Sag Hallo"}' | agentnn dispatch --from-stdin
   ```
   
   Ergebnisse erscheinen als JSON und können per `jq` oder ähnlichen Tools weiterverarbeitet werden.

## Automatisierung mit Templates

Vorlagen in `~/.agentnn/templates/` erlauben es, wiederkehrende Sessions oder Agenten schnell zu starten:

```bash
agentnn template init session --output=my_session.yaml
agentnn quickstart session --from=my_session.yaml
```

Alle Befehle akzeptieren Eingaben über `stdin`, sodass komplexe Workflows scriptbar sind.

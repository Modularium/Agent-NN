# Anwendungsbeispiele

## Chat mit Session und Vector Search

1. Starte einen Session-Manager und den Vector-Store.
2. Füge Dokumente via `/add_document` hinzu.
3. Beginne eine Session mit `/start_session` und sende Fragen an den Dispatcher.

## Search + LLM Workflow

1. Die CLI `agentnn submit "Was ist Agent-NN?"` ruft den Dispatcher auf.
2. Der Worker nutzt den Vector-Store und das LLM-Gateway.
3. Die Antwort wird zusammen mit Quellen zurückgegeben.

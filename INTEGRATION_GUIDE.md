# Integration Guide

Dieser Leitfaden beschreibt die Kopplung von Agent‑NN mit externen Tools.

## Live-Kommunikation mit Flowise-Nodes

Die Datei `api/flowise_bridge.py` stellt einen REST-Router bereit, der Flowise-Flows
direkt mit Agent‑NN interagieren lässt. Über `/flowise/run_task` können Aufgaben
an den Dispatcher geschickt und über `/flowise/status/{task_id}` der Fortschritt
abgerufen werden. Weitere Routen geben Agentinformationen zurück.

Beispielaufruf in Flowise:

```ts
const node = new RunAgentTask('http://localhost:8000', 'Hallo Welt');
const result = await node.run();
```

Die Flowise-Komponente `RunAgentTask` ruft diese Schnittstelle auf und gibt das
API-Ergebnis als Node‑Output zurück.

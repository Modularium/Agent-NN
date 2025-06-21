# Agent-NN Overview

Agent-NN verbindet Agentensteuerung mit neuronalen Netzen und stellt sowohl eine CLI als auch ein SDK bereit. Die Modular Control Plane (MCP) besteht aus mehreren Microservices, die gemeinsam Aufgaben verarbeiten.

## Komponenten

- **Dispatcher** – nimmt Aufgaben entgegen und verteilt sie an Worker-Services
- **Registry** – verwaltet registrierte Agenten und deren Fähigkeiten
- **Gateway** – einheitlicher Zugang zu LLM-Providern
- **VectorStore** – semantische Suche und Dokumentenverwaltung
- **Worker** – spezialisierte Services für Text, Suche und weitere Aufgaben
- **SDK** – Python-Bibliothek für Entwickler
- **CLI** – Kommandozeilenwerkzeug `agentnn`

```mermaid
%% Diagramm folgt in einer späteren Version
```

Weitere Informationen finden sich im Ordner `docs/`.

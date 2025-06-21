# Agent-NN Overview

Agent-NN ist ein modulares Multi-Agent-Framework. Mehrere Microservices bilden zusammen die Modular Control Plane (MCP). Die wichtigsten Komponenten sind:

- **Task Dispatcher** verteilt eingehende Aufgaben.
- **Agent Registry** verwaltet verfügbare Agenten.
- **Session Manager** speichert Gesprächskontexte.
- **Vector Store** ermöglicht semantische Suche.
- **LLM Gateway** stellt eine einheitliche Schnittstelle zu Sprachmodellen bereit.

Die CLI und das SDK erlauben die Interaktion mit diesen Services. Weitere Informationen befinden sich im `docs/` Verzeichnis.

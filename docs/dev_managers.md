# Manager-Übersicht

Dieser Abschnitt gibt eine kurze Übersicht über die wichtigsten Manager im Projekt.

| Manager | Zweck |
|---------|------|
| `AgentManager` | Verwaltet alle WorkerAgents und wählt anhand eines HybridMatchers den passenden Agenten. |
| `EnhancedAgentManager` | Kombiniert AgentOptimizer und NNManager für kontinuierliche Leistungsverbesserung. |
| `AgentOptimizer` | Analysiert Metriken und optimiert Agent-Konfigurationen. |
| `CommunicationManager` | Kümmert sich um die asynchrone Nachrichtenübertragung zwischen Agenten. |
| `ModelManager` | Lädt und registriert verfügbare Modelle. |
| `ModelRegistry` | Versioniert Modelle und speichert Metadaten. |
| `SecurityManager` | Implementiert grundlegende Authentifizierungs- und Prüfmechanismen. |
| `SystemManager` | Übernimmt Systemaufgaben wie Ressourcenprüfung und Wartung. |

Weitere Manager finden sich im Verzeichnis `managers/` und kapseln jeweils ihren Funktionsbereich. Detaillierte Informationen sind in der Code-Dokumentation zu finden.

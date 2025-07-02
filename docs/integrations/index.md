# Integrations

Dieses Kapitel beschreibt, wie Agent-NN mit externen Tools gekoppelt werden kann. Neben der Smolitux-UI stehen Erweiterungen für n8n und FlowiseAI bereit. Beide Systeme können Agent-NN aufrufen und umgekehrt von Agent-NN aus gesteuert werden. Die Plugins unter `plugins/` bringen dazu konfigurierbare HTTP-Parameter mit.

Für die Beispielintegration ist es erforderlich, die TypeScript-Dateien zunächst mit `npm install` und `npx tsc` zu kompilieren. Die erzeugten JavaScript-Dateien werden anschließend vom jeweiligen PluginManager geladen.

Schnelleinstiege findest du in den jeweiligen Kapiteln [n8n](n8n.md#quick-start) und [FlowiseAI](flowise.md#quick-start).

Eine ausführliche Roadmap zur beidseitigen Integration findet sich in [full_integration_plan.md](full_integration_plan.md).

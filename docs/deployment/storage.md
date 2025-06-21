# Persistente Speicherung

Die Microservices speichern Daten standardmäßig im Verzeichnisstruktur unterhalb der in der `.env` definierten Pfade. Alle Verzeichnisse werden beim Start automatisch angelegt.

## Wichtige Umgebungsvariablen

- `DATA_DIR` – Basisverzeichnis für Daten
- `SESSIONS_DIR` – Ablage für Session-Dateien
- `VECTOR_DB_DIR` – Persistenzpfad für den Vector‑Store
- `LOG_DIR` – Logdateien der Services
- `MODELS_DIR` – Lokale Modelle
- `EMBEDDINGS_CACHE_DIR` – Cache für Embeddings
- `EXPORT_DIR` – optionale Exporte

Durch Anpassen dieser Variablen können die Services portable betrieben werden, etwa in Docker-Containern oder auf verschiedenen Hosts.

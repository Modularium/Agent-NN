# Audit Logging

Das System schreibt alle sicherheitsrelevanten Aktionen in JSONL-Dateien unter `audit/`.
Jeder Eintrag enthält Zeitstempel, Akteur, Aktion und Kontext-ID. Dienste fügen ihre
Log-ID im `audit_trace` des jeweiligen `ModelContext` hinzu.

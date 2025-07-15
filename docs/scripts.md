# Script Übersicht

Dieser Abschnitt beschreibt die wichtigsten Shell-Skripte unter `scripts/`.

| Skript | Zweck | Beispiel |
|---|---|---|
| `scripts/setup.sh` | Komplettes Setup ausführen | `./scripts/setup.sh` |
| `scripts/deploy/build_frontend.sh` | React-Frontend bauen | `./scripts/deploy/build_frontend.sh --clean` |
| `scripts/deploy/start_services.sh` | Docker-Services starten | `./scripts/deploy/start_services.sh --build` |
| `scripts/deploy/dev_reset.sh` | Entwicklungsumgebung zurücksetzen | `./scripts/deploy/dev_reset.sh` |
| `scripts/build_and_test.sh` | Docker-Image bauen und Tests ausführen | `./scripts/build_and_test.sh` |
| `scripts/install_dependencies.sh` | System-Abhängigkeiten installieren | `./scripts/install_dependencies.sh --auto-install` |
| `scripts/install/install_packages.sh` | Einzelne Pakete installieren | `./scripts/install/install_packages.sh --with-sudo nodejs npm` |
| `scripts/repair_env.sh` | Umgebung reparieren | `./scripts/repair_env.sh` |

Alle Skripte geben im Fehlerfall einen Exit-Code ungleich Null zurück und unterstützen die Option `--help` für eine Kurzbeschreibung.


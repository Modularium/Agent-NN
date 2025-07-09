#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"

cat <<'EOT'
Agent-NN Befehlsübersicht

  ./scripts/setup.sh         Vollständiges oder teilweises Setup
  ./scripts/install.sh       Abhängigkeiten gezielt installieren
  ./scripts/start_docker.sh  Docker-Services starten
  ./scripts/build_frontend.sh Frontend bauen
  ./scripts/build_and_test.sh Docker-Image bauen und Tests ausführen

Wichtige Flags für setup.sh:
  --full        Komplettes Setup ohne Rückfragen
  --minimal     Nur Python-Abhängigkeiten installieren
  --no-docker   Docker-Schritte überspringen
  --with-docker Setup bricht ab, wenn Docker fehlt

Empfohlene Reihenfolge:
  1. ./scripts/setup.sh --full
  2. ./scripts/start_docker.sh
  3. ./scripts/test.sh
EOT

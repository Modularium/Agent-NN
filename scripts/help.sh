#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"

cat <<'EOT'
Agent-NN Befehlsübersicht

  ./scripts/setup.sh         Interaktives Setup-Menü
  ./scripts/install.sh       Einzelne Komponenten installieren
  ./scripts/start_docker.sh  Docker-Services starten
  ./scripts/build_frontend.sh Frontend bauen
  ./scripts/build_and_test.sh Docker-Image bauen und Tests ausführen

Wichtige Flags für setup.sh:
  --full           Komplettes Setup ohne Rückfragen
  --auto-install   Fehlende Pakete automatisch installieren
  --with-sudo      Systemweite Installation mit sudo
  --no-docker      Docker-Schritte überspringen
  --preset=dev|minimal|ci  Vordefinierte Setups
  --recover        Fehlgeschlagene Schritte überspringen

Empfohlene Reihenfolge:
  1. ./scripts/setup.sh --full
  2. ./scripts/start_docker.sh
  3. ./scripts/test.sh
EOT

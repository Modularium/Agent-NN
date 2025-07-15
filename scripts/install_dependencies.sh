#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/install_utils.sh"
source "$SCRIPT_DIR/lib/preset_utils.sh"

PRESET="dev"

usage() {
    cat <<EOT
Usage: $(basename "$0") [OPTIONS]
Installiere System-Abhängigkeiten.

Optionen:
  --preset NAME    Vordefiniertes Paketset (dev|ci|minimal)
  --auto-install   Keine Rückfragen stellen
  --with-sudo      Paketinstallation mit sudo ausführen
  -h, --help       Diese Hilfe anzeigen
EOT
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"; shift 2;;
        --auto-install)
            AUTO_MODE=true; shift;;
        --with-sudo)
            SUDO_CMD="sudo"; shift;;
        -h|--help)
            usage; exit 0;;
        *)
            echo "Unbekannte Option: $1" >&2; usage; exit 1;;
    esac
done

apply_preset "$PRESET" 2>/dev/null || true

components=()
case "$PRESET" in
    dev)
        components=(ensure_python ensure_poetry ensure_node ensure_docker);
        ;;
    ci)
        components=(ensure_python ensure_poetry);
        ;;
    minimal)
        components=(ensure_python);
        ;;
    *)
        components=(ensure_python ensure_poetry ensure_node ensure_docker);
        ;;
esac

for comp in "${components[@]}"; do
    with_spinner "Installiere ${comp#ensure_}" "$comp"
    status=$?
    [[ $status -eq 0 ]] || echo "${comp#ensure_} fehlgeschlagen" >&2
done

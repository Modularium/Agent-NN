#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/spinner_utils.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"

usage() {
    cat <<EOT
Usage: $(basename "$0") [OPTIONS]
Installiere gezielt Projekt-Abhängigkeiten.

Optionen:
  --docker       Docker und Compose installieren
  --node         Node.js installieren
  --python       Python installieren
  --poetry       Poetry installieren
  --ci           Installiert Python, Poetry, Node und Docker
  -h, --help     Diese Hilfe anzeigen
EOT
}

components=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker) components+=(ensure_docker); shift;;
        --node) components+=(ensure_node); shift;;
        --python) components+=(ensure_python); shift;;
        --poetry) components+=(ensure_poetry); shift;;
        --ci)
            components=(ensure_python ensure_poetry ensure_node ensure_docker)
            shift;;
        -h|--help)
            usage; exit 0;;
        *)
            log_err "Unbekannte Option: $1"
            usage; exit 1;;
    esac
done

if [[ ${#components[@]} -eq 0 ]]; then
    usage
    exit 1
fi

for comp in "${components[@]}"; do
    with_spinner "Installiere ${comp#ensure_}" "$comp" || true
done

exit 0

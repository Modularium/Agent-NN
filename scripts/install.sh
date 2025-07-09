#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/spinner_utils.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"
source "$SCRIPT_DIR/lib/preset_utils.sh"
source "$SCRIPT_DIR/lib/config_utils.sh"
AUTO_MODE=false
LOG_ERROR_FILE="$SCRIPT_DIR/../logs/setup_errors.log"
mkdir -p "$(dirname "$LOG_ERROR_FILE")"
touch "$LOG_ERROR_FILE"
EXIT_ON_FAIL=false
PRESET=""

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
  --exit-on-fail Bei Fehlern sofort abbrechen
  --recover      Installationsschritte überspringen, wenn bereits erledigt
  --preset NAME  Vordefinierte Komponentenauswahl (dev|ci|minimal)
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
            AUTO_MODE=true
            shift;;
        --exit-on-fail)
            EXIT_ON_FAIL=true; shift;;
        --recover)
            RECOVERY_MODE=true; AUTO_MODE=true; shift;;
        --preset)
            shift
            PRESET="$1"
            case "$PRESET" in
                dev)
                    components=(ensure_docker ensure_node ensure_python ensure_poetry)
                    ;;
                ci)
                    components=(ensure_python ensure_poetry)
                    ;;
                minimal)
                    components=(ensure_python)
                    ;;
                *)
                    log_err "Unbekannte Option: $PRESET"; exit 1;;
            esac
            AUTO_MODE=true
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
    with_spinner "Installiere ${comp#ensure_}" "$comp"
    status=$?
    if [[ $status -ne 0 ]]; then
        log_error "${comp#ensure_} konnte nicht installiert werden. Details siehe $LOG_ERROR_FILE"
        [[ "$EXIT_ON_FAIL" == "true" ]] && exit 1
    fi
done

exit 0

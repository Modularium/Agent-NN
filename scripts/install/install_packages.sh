#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/log_utils.sh"
source "$SCRIPT_DIR/../lib/install_utils.sh"

PACKAGES=()
AUTO=false

usage() {
    cat <<EOT
Usage: $(basename "$0") [OPTIONS] PACKAGE [PACKAGE...]
Installiere fehlende System-Pakete.

Optionen:
  --auto-install   Installation ohne Rueckfrage
  --with-sudo      Verwende sudo fuer Paketinstallation
  -h, --help       Diese Hilfe anzeigen
EOT
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-install)
            AUTO=true
            AUTO_MODE=true
            shift;;
        --with-sudo)
            SUDO_CMD="sudo"
            shift;;
        -h|--help)
            usage; exit 0;;
        *)
            PACKAGES+=("$1"); shift;;
    esac
done

if [[ ${#PACKAGES[@]} -eq 0 ]]; then
    usage
    exit 1
fi

install_packages "${PACKAGES[@]}"


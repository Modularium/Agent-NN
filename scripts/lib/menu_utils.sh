#!/bin/bash

__menu_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
}
__menu_utils_init

interactive_menu() {
    PS3="Auswahl: "
    options=(
        "Komplettes Setup (Empfohlen)"
        "Nur Python-Abhängigkeiten installieren"
        "Nur Frontend bauen"
        "Docker-Container starten"
        "Projekt testen"
        "Abbrechen"
    )
    select opt in "${options[@]}"; do
        case $REPLY in
            1) RUN_MODE="full"; break ;;
            2) RUN_MODE="python"; break ;;
            3) RUN_MODE="frontend"; break ;;
            4) RUN_MODE="docker"; break ;;
            5) RUN_MODE="test"; break ;;
            6) exit 0 ;;
            *) echo "Ungültige Auswahl";;
        esac
    done
}

export -f interactive_menu

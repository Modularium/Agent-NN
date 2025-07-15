#!/bin/bash

__menu_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
}
__menu_utils_init

interactive_menu() {
    local options=(
        "💡 Schnellstart" \
        "🧱 Systemabhängigkeiten" \
        "🐍 Python & Poetry" \
        "🎨 Frontend bauen" \
        "🐳 Docker-Komponenten" \
        "🧪 Tests & CI" \
        "🔁 Reparatur" \
        "❌ Abbrechen" \
        "⚙️ Konfiguration anzeigen"
    )
    local count=${#options[@]}
    if command -v whiptail >/dev/null; then
        local choice
        choice=$(whiptail --title "Agent-NN Setup" --menu "Aktion wählen:" 20 78 9 \
            1 "${options[0]}" \
            2 "${options[1]}" \
            3 "${options[2]}" \
            4 "${options[3]}" \
            5 "${options[4]}" \
            6 "${options[5]}" \
            7 "${options[6]}" \
            8 "${options[7]}" \
            9 "${options[8]}" 3>&1 1>&2 2>&3) || exit 1
        case $choice in
            1) RUN_MODE="full" ;;
            2) RUN_MODE="system" ;;
            3) RUN_MODE="python" ;;
            4) RUN_MODE="frontend" ;;
            5) RUN_MODE="docker" ;;
            6) RUN_MODE="test" ;;
            7) RUN_MODE="repair" ;;
            8) RUN_MODE="exit" ;;
            9) RUN_MODE="show_config" ;;
        esac
    else
        PS3="Auswahl: "
        select opt in "${options[@]}"; do
            case $REPLY in
                1) RUN_MODE="full"; break ;;
                2) RUN_MODE="system"; break ;;
                3) RUN_MODE="python"; break ;;
                4) RUN_MODE="frontend"; break ;;
                5) RUN_MODE="docker"; break ;;
                6) RUN_MODE="test"; break ;;
                7) RUN_MODE="repair"; break ;;
                8) RUN_MODE="exit"; break ;;
                9) RUN_MODE="show_config"; break ;;
                *) echo "Ungültige Auswahl";;
            esac
        done
    fi
}

export -f interactive_menu

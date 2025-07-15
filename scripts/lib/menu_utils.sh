#!/bin/bash

__menu_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
}
__menu_utils_init

# Sichere Eingabe für Menü-Auswahl
safe_menu_input() {
    local prompt="$1"
    local timeout="${2:-30}"
    local default="${3:-1}"
    
    if [[ "${AUTO_MODE:-false}" == "true" ]]; then
        echo "$default"
        return 0
    fi
    
    local input=""
    if command -v timeout >/dev/null 2>&1; then
        input=$(timeout "$timeout" bash -c "read -rp '$prompt' input; echo \$input" 2>/dev/null || echo "$default")
    else
        read -rp "$prompt" input 2>/dev/null || input="$default"
    fi
    
    # Fallback auf Default wenn leer
    if [[ -z "$input" ]]; then
        input="$default"
    fi
    
    echo "$input"
}

interactive_menu() {
    local options=(
        "💡 Schnellstart (Vollständige Installation)"
        "🧱 Systemabhängigkeiten (Docker, Node.js, Python)"
        "🐍 Python & Poetry (Nur Python-Umgebung)"
        "🎨 Frontend bauen (React-Frontend)"
        "🐳 Docker-Komponenten (Services starten)"
        "🧪 Tests & CI (Testlauf)"
        "🔁 Reparatur (Umgebung reparieren)"
        "⚙️ Konfiguration anzeigen"
        "🧹 Umgebung bereinigen"
        "❌ Abbrechen"
    )
    
    local count=${#options[@]}
    local choice=""
    local attempts=0
    local max_attempts=3
    
    while [[ $attempts -lt $max_attempts ]]; do
        echo "╔══════════════════════════════════════════════════════════════════════════════╗"
        echo "║                              Agent-NN Setup                                 ║"
        echo "╠══════════════════════════════════════════════════════════════════════════════╣"
        echo "║  Wähle eine Aktion:                                                         ║"
        echo "╠══════════════════════════════════════════════════════════════════════════════╣"
        
        for i in "${!options[@]}"; do
            printf "║  [%d] %-70s ║\n" "$((i + 1))" "${options[$i]}"
        done
        
        echo "╚══════════════════════════════════════════════════════════════════════════════╝"
        echo
        
        # Prüfe ob whiptail verfügbar ist
        if command -v whiptail >/dev/null 2>&1 && [[ -t 0 ]] && [[ -t 1 ]]; then
            local menu_items=()
            for i in "${!options[@]}"; do
                menu_items+=("$((i + 1))" "${options[$i]}")
            done
            
            if choice=$(whiptail --title "Agent-NN Setup" --menu "Aktion wählen:" 20 78 10 "${menu_items[@]}" 3>&1 1>&2 2>&3); then
                case $choice in
                    1) RUN_MODE="full" ;;
                    2) RUN_MODE="system" ;;
                    3) RUN_MODE="python" ;;
                    4) RUN_MODE="frontend" ;;
                    5) RUN_MODE="docker" ;;
                    6) RUN_MODE="test" ;;
                    7) RUN_MODE="repair" ;;
                    8) RUN_MODE="show_config" ;;
                    9) RUN_MODE="clean" ;;
                    10) RUN_MODE="exit" ;;
                    *) RUN_MODE="exit" ;;
                esac
                return 0
            else
                RUN_MODE="exit"
                return 0
            fi
        else
            # Fallback zu normaler Eingabe
            choice=$(safe_menu_input "Auswahl [1-${count}]: " 30 "1")
            
            case $choice in
                1) RUN_MODE="full"; break ;;
                2) RUN_MODE="system"; break ;;
                3) RUN_MODE="python"; break ;;
                4) RUN_MODE="frontend"; break ;;
                5) RUN_MODE="docker"; break ;;
                6) RUN_MODE="test"; break ;;
                7) RUN_MODE="repair"; break ;;
                8) RUN_MODE="show_config"; break ;;
                9) RUN_MODE="clean"; break ;;
                10) RUN_MODE="exit"; break ;;
                ""|q|Q) RUN_MODE="exit"; break ;;
                *)
                    attempts=$((attempts + 1))
                    log_warn "Ungültige Auswahl: $choice"
                    if [[ $attempts -ge $max_attempts ]]; then
                        log_warn "Zu viele ungültige Eingaben. Verwende Schnellstart."
                        RUN_MODE="full"
                        break
                    fi
                    echo "Bitte wähle eine Zahl zwischen 1 und $count."
                    echo "Drücke Enter für Schnellstart oder q zum Beenden."
                    sleep 2
                    ;;
            esac
        fi
    done
    
    if [[ -z "$RUN_MODE" ]]; then
        RUN_MODE="full"
    fi
    
    log_info "Gewählte Aktion: $RUN_MODE"
}

# Bestätigungsmenü für kritische Aktionen
confirm_action() {
    local action="$1"
    local message="${2:-Möchten Sie fortfahren?}"
    
    if [[ "${AUTO_MODE:-false}" == "true" ]]; then
        return 0
    fi
    
    echo "⚠️  $message"
    echo "Aktion: $action"
    echo
    
    local choice
    choice=$(safe_menu_input "Fortfahren? [j/N]: " 15 "n")
    
    case "${choice,,}" in
        j|ja|y|yes) return 0 ;;
        *) return 1 ;;
    esac
}

# Auswahlmenü für Optionen
select_option() {
    local title="$1"
    shift
    local options=("$@")
    local count=${#options[@]}
    
    echo "=== $title ==="
    echo
    
    for i in "${!options[@]}"; do
        echo "  [$((i + 1))] ${options[$i]}"
    done
    echo
    
    local choice
    choice=$(safe_menu_input "Auswahl [1-$count]: " 30 "1")
    
    if [[ "$choice" -ge 1 && "$choice" -le "$count" ]]; then
        echo "${options[$((choice - 1))]}"
        return 0
    else
        echo "${options[0]}"  # Default zur ersten Option
        return 1
    fi
}

export -f interactive_menu confirm_action select_option safe_menu_input

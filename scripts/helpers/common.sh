#!/bin/bash
# -*- coding: utf-8 -*-

# Prevent multiple sourcing
if [[ "${_COMMON_SH_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly _COMMON_SH_LOADED=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/log_utils.sh"

# Utility-Funktionen
check_command() {
    local cmd="$1"
    local name="${2:-$cmd}"
    
    if command -v "$cmd" &>/dev/null; then
        log_debug "$name ist verfügbar: $(command -v "$cmd")"
        return 0
    else
        log_err "$name ist nicht installiert oder nicht im PATH"
        return 1
    fi
}

check_port() {
    local port="$1"
    if command -v lsof &>/dev/null; then
        if lsof -i ":$port" &>/dev/null; then
            log_warn "Port $port ist bereits belegt"
            return 1
        fi
    elif command -v netstat &>/dev/null; then
        if netstat -ln | grep -q ":$port "; then
            log_warn "Port $port ist bereits belegt"
            return 1
        fi
    elif command -v ss &>/dev/null; then
        if ss -ln | grep -q ":$port "; then
            log_warn "Port $port ist bereits belegt"
            return 1
        fi
    else
        log_debug "Keine Tool für Port-Prüfung verfügbar"
    fi
    return 0
}

# Fehlerbehandlung einrichten
setup_error_handling() {
    set -euo pipefail
    trap 'log_err "Fehler in Zeile $LINENO. Exit-Code: $?"' ERR
}

# UTF-8 Locale sicherstellen
ensure_utf8() {
    if [[ "${LC_ALL:-}" != *UTF-8* ]] && [[ "${LANG:-}" != *UTF-8* ]]; then
        export LANG="${LANG:-en_US.UTF-8}"
        export LC_ALL="${LC_ALL:-en_US.UTF-8}"
        log_debug "UTF-8 Locale eingestellt: LANG=$LANG, LC_ALL=$LC_ALL"
    fi
}

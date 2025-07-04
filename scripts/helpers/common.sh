#!/bin/bash
# -*- coding: utf-8 -*-

# Farb-Codes für konsistente Ausgabe
readonly RED='\033[1;31m'
readonly GREEN='\033[1;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[1;34m'
readonly PURPLE='\033[1;35m'
readonly CYAN='\033[1;36m'
readonly NC='\033[0m' # No Color

# Logging-Funktionen mit UTF-8 Unterstützung
log_info() { 
    echo -e "${BLUE}[...]${NC} $1" 
}

log_ok() { 
    echo -e "${GREEN}[✓]${NC} $1" 
}

log_warn() { 
    echo -e "${YELLOW}[⚠]${NC} $1" 
}

log_err() { 
    echo -e "${RED}[✗]${NC} $1" >&2 
}

log_debug() {
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1" >&2
    fi
}

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

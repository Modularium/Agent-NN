#!/bin/bash
# -*- coding: utf-8 -*-

set -euo pipefail

# Only set HELPERS_DIR if it's not already set
if [[ -z "${HELPERS_DIR:-}" ]]; then
    HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Source common.sh from the helpers directory
source "$HELPERS_DIR/common.sh"

# Global array to hold missing packages when running checks
MISSING_PACKAGES=()

check_env_file() {
    local env_file="${1:-.env}"
    local example_file="${2:-.env.example}"
    
    log_info "Prüfe Umgebungskonfiguration..."
    
    if [[ ! -f "$env_file" ]]; then
        if [[ -f "$example_file" ]]; then
            log_warn "$env_file nicht gefunden. Erstelle aus $example_file..."
            if cp "$example_file" "$env_file"; then
                log_ok "$env_file erstellt"
                log_warn "Bitte überprüfe und passe die Werte in $env_file an!"
            else
                log_err "Konnte $env_file nicht erstellen"
                return 1
            fi
        else
            log_err "Weder $env_file noch $example_file gefunden"
            return 1
        fi
    else
        log_ok "$env_file gefunden"
    fi
    
    # Grundlegende Validierung der .env-Datei
    if [[ -f "$env_file" ]]; then
        local required_vars=("DATABASE_URL" "LLM_BACKEND")
        local missing_vars=()
        
        for var in "${required_vars[@]}"; do
            if ! grep -q "^${var}=" "$env_file"; then
                missing_vars+=("$var")
            fi
        done
        
        if [[ ${#missing_vars[@]} -gt 0 ]]; then
            log_warn "Fehlende Variablen in $env_file: ${missing_vars[*]}"
            log_warn "Vergleiche mit $example_file"
        fi
    fi
    
    return 0
}

check_system_dependencies() {
    log_info "Prüfe System-Abhängigkeiten..."
    
    local missing_deps=()
    local deps=(
        "python3:Python 3"
        "pip:pip"
        "curl:curl"
        "node:Node.js"
        "npm:npm"
        "git:Git"
    )
    
    for dep in "${deps[@]}"; do
        local cmd="${dep%:*}"
        local name="${dep#*:}"
        
        if ! check_command "$cmd" "$name"; then
            missing_deps+=("$name")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_err "Fehlende Abhängigkeiten: ${missing_deps[*]}"
        log_err "Installationshinweise:"
        log_err "  Ubuntu/Debian: sudo apt update && sudo apt install python3-pip curl git nodejs npm"
        log_err "  macOS: brew install python pip curl git node npm"
        log_err "  Windows: Verwende WSL oder installiere manuell"
        echo "${missing_deps[@]}"
        return 1
    fi

    log_ok "Alle System-Abhängigkeiten verfügbar"
    return 0
}

check_python_environment() {
    log_info "Prüfe Python-Umgebung..."

    # Python-Version prüfen
    local python_version
    local missing=()
    if python_version=$(python3 --version 2>/dev/null); then
        log_debug "Python gefunden: $python_version"

        # Mindestversion prüfen (3.9+)
        local version_number
        version_number=$(echo "$python_version" | sed 's/Python //' | cut -d. -f1,2)
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
            log_ok "Python $version_number ist kompatibel"
        else
            log_err "Python $version_number ist zu alt. Mindestens Python 3.9 erforderlich."
            missing+=("python3")
        fi
    else
        log_err "Python 3 nicht gefunden"
        missing+=("python3")
    fi

    if ! command -v pip >/dev/null; then
        log_err "pip nicht gefunden"
        missing+=("pip")
    fi
    
    # Poetry prüfen
    if check_command poetry "Poetry"; then
        local poetry_version
        poetry_version=$(poetry --version 2>/dev/null | cut -d' ' -f3 || echo "unknown")
        log_ok "Poetry verfügbar (v$poetry_version)"
    else
        log_err "Poetry nicht gefunden. Installation:"
        log_err "  curl -sSL https://install.python-poetry.org | python3 -"
        log_err "  Oder: pip install poetry"
        missing+=("poetry")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "${missing[@]}"
        return 1
    fi

    return 0
}

check_node_environment() {
    log_info "Prüfe Node.js-Umgebung..."

    # Node.js Version prüfen
    local node_version
    local missing=()
    if node_version=$(node --version 2>/dev/null); then
        log_debug "Node.js gefunden: $node_version"

        # Mindestversion prüfen (18+)
        local major_version
        major_version=$(echo "$node_version" | sed 's/v//' | cut -d. -f1)
        if [[ "$major_version" -ge 18 ]]; then
            log_ok "Node.js $node_version ist kompatibel"
        else
            log_err "Node.js $node_version ist zu alt. Mindestens Node.js 18 erforderlich."
            missing+=("nodejs")
        fi
    else
        log_err "Node.js nicht gefunden"
        missing+=("nodejs")
    fi

    # npm prüfen
    if check_command npm "npm"; then
        local npm_version
        npm_version=$(npm --version 2>/dev/null || echo "unknown")
        log_ok "npm verfügbar (v$npm_version)"
    else
        log_err "npm nicht gefunden"
        missing+=("npm")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "${missing[@]}"
        return 1
    fi

    return 0
}

check_ports() {
    log_info "Prüfe Port-Verfügbarkeit..."
    
    local ports=(8000 3000 5432 6379 9090)
    local blocked_ports=()
    
    for port in "${ports[@]}"; do
        if ! check_port "$port"; then
            blocked_ports+=("$port")
        fi
    done
    
    if [[ ${#blocked_ports[@]} -gt 0 ]]; then
        log_warn "Belegte Ports: ${blocked_ports[*]}"
        log_warn "Diese Ports werden von Agent-NN benötigt:"
        log_warn "  8000: API Gateway"
        log_warn "  3000: Frontend"
        log_warn "  5432: PostgreSQL"
        log_warn "  6379: Redis"
        log_warn "  9090: Prometheus"
        log_warn "Stoppe andere Services oder ändere die Ports in der docker-compose.yml"
    else
        log_ok "Alle benötigten Ports sind verfügbar"
    fi
    
    return 0
}

# Haupt-Umgebungsprüfung
check_environment() {
    log_info "Führe vollständige Umgebungsprüfung durch..."
    ensure_utf8

    local checks=(
        check_system_dependencies
        check_python_environment
        check_node_environment
        check_env_file
        check_ports
    )

    local failed_checks=()
    local missing=()

    for check in "${checks[@]}"; do
        local out=""
        if out=$("$check" 2>&1); then
            :
        else
            failed_checks+=("$check")
            [[ -n "$out" ]] && missing+=( $out )
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        MISSING_PACKAGES=($(printf "%s\n" "${missing[@]}" | sort -u))
    else
        MISSING_PACKAGES=()
    fi

    if [[ ${#failed_checks[@]} -gt 0 ]]; then
        log_err "Umgebungsprüfung fehlgeschlagen: ${failed_checks[*]}"
        echo "${MISSING_PACKAGES[@]}"
        return 1
    fi

    log_ok "Umgebungsprüfung erfolgreich abgeschlossen"
    return 0
}

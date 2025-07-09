#!/bin/bash

__env_check_init() {
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/../helpers/common.sh"
}

__env_check_init

check_dotenv_file() {
    local env_file="${1:-.env}"
    local example_file="${2:-.env.example}"

    if [[ ! -f "$env_file" ]]; then
        if [[ -f "$example_file" ]]; then
            log_warn "$env_file fehlt – erstelle aus $example_file"
            cp "$example_file" "$env_file"
        else
            log_err "$env_file und Vorlage $example_file fehlen"
            return 1
        fi
    else
        log_ok "$env_file vorhanden"
    fi
    return 0
}

check_ports() {
    local ports=("$@")
    local blocked=()
    for p in "${ports[@]}"; do
        if ! check_port "$p"; then
            blocked+=("$p")
        fi
    done
    if [[ ${#blocked[@]} -gt 0 ]]; then
        log_warn "Ports belegt: ${blocked[*]}"
        return 1
    fi
    log_ok "Alle Ports frei"
    return 0
}

log_env_status() {
    log_info "Aktuelle .env Einstellungen:" && grep -v '^#' .env 2>/dev/null || true
}

env_check() {
    check_dotenv_file && check_ports 8000 3000 5432 6379 9090
}

export -f check_dotenv_file check_ports log_env_status env_check

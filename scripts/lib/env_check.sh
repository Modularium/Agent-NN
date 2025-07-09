#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../helpers/common.sh"

check_env_file() {
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
    return 0
}

env_check() {
    check_env_file && check_ports 8000 3000 5432 6379 9090
}

export -f check_env_file check_ports env_check

#!/bin/bash

__install_utils_init() {
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/../helpers/common.sh"
}

__install_utils_init

AUTO_MODE="${AUTO_MODE:-false}"

ask_install() {
    local component="$1"
    if [[ "$AUTO_MODE" == "true" ]]; then
        return 0
    fi
    read -rp "Fehlende Komponente $component gefunden. Jetzt installieren? [J/n] " ans
    if [[ -z "$ans" || "$ans" =~ ^[JjYy]$ ]]; then
        return 0
    fi
    return 1
}

install_docker() {
    log_info "Installiere Docker..."
    curl -fsSL https://get.docker.com | sh >/dev/null
}

ensure_docker() {
    if ! command -v docker &>/dev/null; then
        if ask_install "Docker"; then
            install_docker || return 1
        else
            return 1
        fi
    fi
    if ! docker compose version &>/dev/null && ! command -v docker-compose &>/dev/null; then
        log_warn "Docker Compose fehlt"
        # Docker install script includes compose plugin
    fi
    return 0
}

install_node() {
    log_info "Installiere Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - >/dev/null && apt-get install -y nodejs >/dev/null
}

ensure_node() {
    if ! command -v node &>/dev/null; then
        if ask_install "Node.js"; then
            install_node || return 1
        else
            return 1
        fi
    fi
    command -v npm &>/dev/null || apt-get install -y npm >/dev/null
    return 0
}

install_python() {
    log_info "Installiere Python 3.10..."
    apt-get update -y >/dev/null && apt-get install -y python3.10 python3.10-venv python3.10-distutils >/dev/null
}

ensure_python() {
    local version
    if version=$(python3 --version 2>/dev/null); then
        if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)' ; then
            if ask_install "Python 3.10"; then
                install_python || return 1
            else
                return 1
            fi
        fi
    else
        if ask_install "Python 3.10"; then
            install_python || return 1
        else
            return 1
        fi
    fi
}

install_poetry() {
    log_info "Installiere Poetry..."
    pipx install poetry >/dev/null 2>&1 || pip install poetry >/dev/null
}

ensure_poetry() {
    if ! command -v poetry &>/dev/null; then
        if ask_install "Poetry"; then
            install_poetry || return 1
        else
            return 1
        fi
    fi
}

install_python_tool() {
    local tool="$1"
    pip install "$tool" >/dev/null
}

ensure_python_tools() {
    local tools=(ruff mypy pytest)
    for t in "${tools[@]}"; do
        if ! command -v "$t" &>/dev/null; then
            if ask_install "$t"; then
                install_python_tool "$t" || return 1
            fi
        fi
    done
}

export -f ask_install install_docker ensure_docker install_node ensure_node install_python ensure_python install_poetry ensure_poetry install_python_tool ensure_python_tools

show_spinner() {
    local pid=$1
    local delay=0.1
    local spin='|/-\\'
    while kill -0 "$pid" 2>/dev/null; do
        for i in $spin; do
            printf "\r[%s] $SPINNER_MSG" "$i"
            sleep $delay
        done
    done
    wait "$pid" 2>/dev/null
    printf "\r"
}

with_spinner() {
    SPINNER_MSG="$1"; shift
    ("$@") &
    local pid=$!
    show_spinner $pid
    local status=$?
    return $status
}

export -f show_spinner with_spinner

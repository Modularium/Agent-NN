#!/bin/bash

__frontend_build_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
    source "$dir/../helpers/common.sh"
    source "$dir/status_utils.sh"
}

__frontend_build_init

build_frontend() {
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local dir="$SCRIPT_DIR/../../frontend/agent-ui"
    local target="$SCRIPT_DIR/../../frontend/dist"

    if [[ ! -d "$dir" ]]; then
        log_err "Frontend-Verzeichnis fehlt"
        return 1
    fi

    pushd "$dir" >/dev/null || return 1
    if [[ ! -d node_modules ]]; then
        log_info "Installiere Frontend-Abhängigkeiten"
        npm ci || npm install || return 1
    fi

    log_info "Baue Frontend"
    npm run build || return 1
    popd >/dev/null

    mkdir -p "$target"
    cp -r "$dir"/dist/* "$target"/ 2>/dev/null || true
    log_info "Build-Output:"
    ls -al "$target" || true
    log_ok "Frontend-Build abgeschlossen"
    update_status "frontend" "built" "$SCRIPT_DIR/../../.agentnn/status.json"
}

export -f build_frontend

#!/bin/bash

function build_frontend() {
    local dir="frontend/agent-ui"
    if [[ ! -d "$dir" ]]; then
        echo "Frontend-Verzeichnis fehlt" >&2
        return 1
    fi
    pushd "$dir" >/dev/null || return 1
    if [[ ! -d node_modules ]]; then
        npm ci || npm install || return 1
    fi
    npm run build || return 1
    popd >/dev/null
    mkdir -p frontend/dist
    cp -r "$dir"/dist/* frontend/dist/ 2>/dev/null || true
}

#!/bin/bash

function env_check() {
    local missing=()
    for cmd in python3 node npm git; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Fehlende Abhängigkeiten: ${missing[*]}" >&2
        return 1
    fi
    [[ -f .env ]] || cp .env.example .env 2>/dev/null || true
}

#!/bin/bash

# Preset definitions used by setup and install scripts.
#
# Available presets:
#   dev     - full installation including Docker and frontend build
#   ci      - dependencies for running the test-suite only
#   minimal - Python environment without Docker or Node.js

declare -A PRESETS=(
    [minimal]="core only"
    [dev]="core + frontend + langchain"
    [ci]="core + test tools + headless frontend"
)

__preset_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
}
__preset_utils_init

validate_preset() {
    local preset="$1"
    if [[ -z "${PRESETS[$preset]+x}" ]]; then
        log_err "Unbekanntes Preset: $preset"
        return 1
    fi
}

apply_preset() {
    local preset="$1"
    validate_preset "$preset" || return 1
    PRESET="$preset"
    case "$preset" in
        dev)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            ;;
        ci)
            RUN_MODE="test"
            BUILD_FRONTEND=false
            START_DOCKER=false
            ;;
        minimal)
            RUN_MODE="python"
            BUILD_FRONTEND=false
            START_DOCKER=false
            ;;
    esac
}

export -f apply_preset validate_preset

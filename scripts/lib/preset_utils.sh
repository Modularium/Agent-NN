#!/bin/bash
# -*- coding: utf-8 -*-
# Preset definitions used by setup and install scripts.

# Associative array mapping preset names to descriptions.
declare -A PRESETS=(
    [minimal]="Minimale Python-Umgebung"
    [dev]="Vollstaendige Entwicklungsumgebung"
    [ci]="CI/CD Pipeline Setup"
    [mcp]="MCP Services Setup"
    [full]="Komplettes System mit allen Features"
)

# Initialize logging utilities from the same directory.
__preset_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
}
__preset_utils_init

# Validate that the given preset exists.
validate_preset() {
    local preset="$1"
    if [[ -z "${PRESETS[$preset]+x}" ]]; then
        log_err "Unbekanntes Preset: $preset"
        log_info "Verfuegbare Presets:"
        for p in "${!PRESETS[@]}"; do
            log_info "  $p - ${PRESETS[$p]}"
        done
        return 1
    fi
}

# Apply variables according to preset.
apply_preset() {
    local preset="$1"
    validate_preset "$preset" || return 1

    PRESET="$preset"
    export PRESET

    case "$preset" in
        minimal)
            RUN_MODE="python"
            BUILD_FRONTEND=false
            START_DOCKER=false
            START_MCP=false
            ;;
        dev)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            START_MCP=false
            ;;
        ci)
            RUN_MODE="test"
            BUILD_FRONTEND=false
            START_DOCKER=false
            START_MCP=false
            ;;
        mcp)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            START_MCP=true
            ;;
        full)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            START_MCP=true
            ;;
    esac

    export RUN_MODE BUILD_FRONTEND START_DOCKER START_MCP
}

export -f validate_preset apply_preset

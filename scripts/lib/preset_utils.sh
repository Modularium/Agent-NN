#!/bin/bash

__preset_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
}
__preset_utils_init

apply_preset() {
    local preset="$1"
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
        *)
            log_warn "Unknown preset: $preset"
            return 1
            ;;
    esac
}

export -f apply_preset

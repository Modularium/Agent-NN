#!/bin/bash

# Logging utilities for scripts

# initialization
__log_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
}
__log_utils_init

# color codes (if not already defined)
if [[ -z "${RED:-}" ]]; then
    readonly RED='\033[1;31m'
    readonly GREEN='\033[1;32m'
    readonly YELLOW='\033[1;33m'
    readonly BLUE='\033[1;34m'
    readonly PURPLE='\033[1;35m'
    readonly CYAN='\033[1;36m'
    readonly NC='\033[0m'
fi

log_info() {
    echo -e "${BLUE}[...]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_err() {
    echo -e "${RED}[✗]${NC} $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1" >&2
    fi
}

export -f log_info log_ok log_warn log_err log_debug

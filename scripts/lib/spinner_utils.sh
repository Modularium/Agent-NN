#!/bin/bash

__spinner_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
}
__spinner_utils_init

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

#!/bin/bash

__spinner_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
}
__spinner_utils_init

show_spinner() {
    local msg="$1"; shift
    ("$@") &
    local pid=$!
    local i=0
    local spinners=("⠋" "⠙" "⠸" "⠴" "⠦" "⠧" "⠇" "⠏")
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r%s %s" "${spinners[i]}" "$msg"
        i=$(( (i + 1) % ${#spinners[@]} ))
        sleep 0.1
    done
    wait "$pid" 2>/dev/null
    printf "\r"
    return $?
}

with_spinner() {
    show_spinner "$@"
}

export -f show_spinner with_spinner

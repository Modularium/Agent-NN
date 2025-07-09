#!/bin/bash

__args_parser_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
}
__args_parser_init

parse_setup_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                export DEBUG=1
                ;;
            --no-frontend)
                BUILD_FRONTEND=false
                ;;
            --skip-docker)
                START_DOCKER=false
                ;;
            --check-only)
                BUILD_FRONTEND=false
                START_DOCKER=false
                ;;
            --check)
                RUN_MODE="check"
                BUILD_FRONTEND=false
                START_DOCKER=false
                ;;
            --install-heavy)
                INSTALL_HEAVY=true
                ;;
            --with-docker)
                WITH_DOCKER=true
                ;;
            --full)
                AUTO_MODE=true
                RUN_MODE="full"
                ;;
            --minimal)
                START_DOCKER=false
                BUILD_FRONTEND=false
                AUTO_MODE=true
                RUN_MODE="python"
                ;;
            --no-docker)
                START_DOCKER=false
                ;;
            --exit-on-fail)
                EXIT_ON_FAIL=true
                ;;
            --recover)
                RECOVERY_MODE=true
                AUTO_MODE=true
                ;;
            --preset)
                shift
                PRESET="$1"
                apply_preset "$PRESET"
                ;;
            --clean)
                clean_environment
                exit 0
                ;;
            *)
                log_err "Unbekannte Option: $1"
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

export -f parse_setup_args

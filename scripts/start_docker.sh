#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers/common.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"

LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/start_docker.log"
exec > >(tee -a "$LOG_FILE") 2>&1

usage() {
    echo "Usage: $(basename "$0") [--env NAME]"
    echo "Startet Docker Compose mit optionaler Umgebung"
}

ENV_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_NAME="$2"; shift 2;;
        -h|--help)
            usage; exit 0;;
        *)
            shift;;
    esac
done

compose_file="docker-compose${ENV_NAME:+.$ENV_NAME}.yml"

if ! has_docker || ! has_docker_compose; then
    log_err "Docker oder Docker Compose fehlt"
    exit 1
fi

if [[ ! -f "$compose_file" ]]; then
    log_warn "Compose-Datei $compose_file nicht gefunden"
    exit 0
fi

start_compose "$compose_file"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/docker_utils.sh"

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

if ! has_docker; then
    echo "Docker nicht gefunden" >&2
    exit 1
fi

if [[ ! -f "$compose_file" ]]; then
    echo "Compose-Datei $compose_file nicht gefunden" >&2
    exit 1
fi

start_compose "$compose_file"

#!/bin/bash
# -*- coding: utf-8 -*-

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers/docker.sh"

# Start Docker services using the docker_compose_up function
compose_file=$(find_compose_file "docker-compose.yml") || exit 1
docker_compose_up "$compose_file" "--build" "-d"

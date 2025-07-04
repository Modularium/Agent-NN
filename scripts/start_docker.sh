#!/bin/bash
# -*- coding: utf-8 -*-

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers/docker.sh"

# Start Docker services using the docker_compose_up function
docker_compose_up "docker-compose.yml" "--build" "-d"

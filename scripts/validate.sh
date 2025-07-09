#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"

log_info "Starte Validierung..."

errors=0

check_dotenv_file || errors=$((errors+1))
check_ports 8000 3000 5432 6379 9090 || errors=$((errors+1))

if ! has_docker || ! docker ps > /dev/null 2>&1; then
    log_err "Docker Container laufen nicht"
    errors=$((errors+1))
fi

ensure_python || errors=$((errors+1))
ensure_python_tools || errors=$((errors+1))

if [[ $errors -eq 0 ]]; then
    log_ok "Validierung erfolgreich"
else
    log_warn "Validierung fand $errors Probleme"
fi

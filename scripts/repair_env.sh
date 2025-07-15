#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"

log_info "Prüfe Umgebung..."
if ! check_environment; then
    log_warn "Versuche fehlende Komponenten zu installieren"
    ensure_python || true
    ensure_poetry || true
    ensure_node || true
    ensure_docker || true
    check_dotenv_file || true
fi

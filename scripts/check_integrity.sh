#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/lib/log_utils.sh"

errors=0
required=(".env.example" "docker-compose.yml" "frontend" "pyproject.toml")
for f in "${required[@]}"; do
    if [[ ! -e "$REPO_ROOT/$f" ]]; then
        log_err "$f fehlt"
        errors=$((errors+1))
    fi
done

if git diff --name-only --exit-code >/dev/null; then
    :
else
    log_warn "Uncommitted changes erkannt"
fi

if [[ $errors -eq 0 ]]; then
    log_ok "Integritätsprüfung erfolgreich"
else
    log_warn "Integritätsprüfung fand $errors Probleme"
fi

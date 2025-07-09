#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STATUS_FILE="$REPO_ROOT/.agentnn/status.json"

source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"

print_last_setup() {
    if [[ -f "$STATUS_FILE" ]]; then
        python - <<PY "$STATUS_FILE"
import json,sys
s=json.load(open(sys.argv[1]))
print("Last setup:", s.get("last_setup", "unknown"))
for k,v in s.items():
    if k != "last_setup":
        print(f"{k}: {v}")
PY
    else
        log_warn "Status file not found"
    fi
}

check_consistency() {
    local issues=()
    if [[ -f "$REPO_ROOT/frontend/dist/index.html" && ! -f "$STATUS_FILE" ]]; then
        issues+=("Frontend built but no status file")
    fi
    if [[ -f "$STATUS_FILE" ]]; then
        local docker_state
        docker_state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("docker",""))' "$STATUS_FILE")
        if [[ "$docker_state" == "ok" ]] && ! docker ps >/dev/null 2>&1; then
            issues+=("Docker status ok but services not running")
        fi
    fi
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_warn "Inconsistent state detected: ${issues[*]}"
    else
        log_ok "Status consistent"
    fi
}

main() {
    log_info "Setup status:"
    print_last_setup
    check_consistency
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

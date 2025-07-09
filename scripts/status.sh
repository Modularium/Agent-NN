#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STATUS_FILE="$REPO_ROOT/.agentnn/status.json"

source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
source "$SCRIPT_DIR/lib/status_utils.sh"

ensure_status_file "$STATUS_FILE"

get_status_value() {
    python - "$STATUS_FILE" "$1" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    data=json.load(f)
print(data.get(sys.argv[2], ""))
PY
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
        return 1
    else
        log_ok "Status consistent"
        return 0
    fi
}

main() {
    local last_setup preset consistency
    last_setup=$(get_status_value last_setup)
    preset=$(get_status_value preset)
    if check_consistency >/dev/null; then
        consistency="OK"
    else
        consistency="FEHLER"
    fi
    log_last_check "$STATUS_FILE"
    echo "Letztes Setup:   ${last_setup:-unknown}"
    echo "Preset genutzt:  ${preset:--}"
    echo "Konsistenz:      $consistency"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

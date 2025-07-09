#!/bin/bash

__status_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    STATUS_UTILS_DIR="$dir"
    source "$dir/log_utils.sh"
}
__status_utils_init

STATUS_FILE="${STATUS_FILE:-}" # allow override

load_status() {
    local file="${1:-$STATUS_FILE}"
    [[ -f "$file" ]] || return 1
    cat "$file"
}

ensure_status_file() {
    local file="${1:-$STATUS_FILE}"
    mkdir -p "$(dirname "$file")"
    [[ -f "$file" ]] || echo '{}' > "$file"
}

log_last_setup() {
    local file="${1:-$STATUS_FILE}"
    update_status "last_setup" "$(date -u +%FT%TZ)" "$file"
}

log_last_check() {
    local file="${1:-$STATUS_FILE}"
    update_status "last_check" "$(date -u +%FT%TZ)" "$file"
}

log_preset() {
    local preset="$1"
    local file="${2:-$STATUS_FILE}"
    update_status "preset" "$preset" "$file"
}

update_status() {
    local key="$1"; local value="$2"
    local file="${3:-$STATUS_FILE}"
    ensure_status_file "$file"
    python - "$file" "$key" "$value" <<'PY'
import json,sys
path,key,val=sys.argv[1:]
with open(path) as f:
    data=json.load(f)
if val == "":
    data.pop(key, None)
else:
    data[key]=val
with open(path,'w') as f:
    json.dump(data,f)
PY
}

export -f load_status update_status ensure_status_file \
        log_last_setup log_last_check log_preset

#!/bin/bash

__status_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
}
__status_utils_init

STATUS_FILE="${STATUS_FILE:-}" # allow override

load_status() {
    local file="${1:-$STATUS_FILE}"
    [[ -f "$file" ]] || return 1
    cat "$file"
}

update_status() {
    local key="$1"; local value="$2"
    local file="${3:-$STATUS_FILE}"
    mkdir -p "$(dirname "$file")"
    if [[ ! -f "$file" ]]; then
        echo '{}' > "$file"
    fi
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

export -f load_status update_status

#!/bin/bash

__config_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
}
__config_utils_init

load_config() {
    local config_file="$HOME/.agentnn/config.json"
    if [[ ! -f "$config_file" ]]; then
        return 0
    fi

    DEFAULT_MODE=$(python - <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data.get("default_mode", ""))
PY
"$config_file")

    LAST_USED_FLAGS=$(python - <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
flags = data.get("last_used_flags", [])
print(" ".join(flags))
PY
"$config_file")

    PYTHON_VERSION_PREF=$(python - <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data.get("python_version", ""))
PY
"$config_file")

    PROJECT_PATH_PREF=$(python - <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data.get("project_path", ""))
PY
"$config_file")
}

export -f load_config

# Project-specific config stored in REPO_ROOT/.agentnn_config
PROJECT_CONFIG_FILE="${PROJECT_CONFIG_FILE:-$REPO_ROOT/.agentnn_config}"

ensure_config_file_exists() {
    mkdir -p "$(dirname "$PROJECT_CONFIG_FILE")"
    if [ ! -f "$PROJECT_CONFIG_FILE" ]; then
        echo "# Agent-NN Setup Config" > "$PROJECT_CONFIG_FILE"
    fi
}

load_project_config() {
    ensure_config_file_exists
    [[ -f "$PROJECT_CONFIG_FILE" ]] && source "$PROJECT_CONFIG_FILE"
}

get_config_value() {
    local key="$1"
    grep -E "^${key}=" "$PROJECT_CONFIG_FILE" 2>/dev/null | head -n1 | cut -d= -f2- | tr -d '"'
}

save_config_value() {
    local key="$1"; local value="${2:-}"
    ensure_config_file_exists
    [[ -z "$key" || -z "$value" ]] && return 0
    if grep -q "^${key}=" "$PROJECT_CONFIG_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=\"${value}\"|" "$PROJECT_CONFIG_FILE"
    else
        echo "${key}=\"${value}\"" >> "$PROJECT_CONFIG_FILE"
    fi
}

load_config_value() {
    local key="$1"; local default="$2"
    ensure_config_file_exists
    local value
    value=$(grep -E "^${key}=" "$PROJECT_CONFIG_FILE" 2>/dev/null | head -n1 | cut -d= -f2- | tr -d '"')
    if [[ -z "$value" ]]; then
        log_info "→ Kein gespeicherter Wert für ${key,,}, benutze Default: $default"
        echo "$default"
    else
        echo "$value"
    fi
}

export -f load_project_config get_config_value \
          save_config_value load_config_value ensure_config_file_exists

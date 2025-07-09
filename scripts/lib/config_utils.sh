#!/bin/bash

__config_utils_init() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/log_utils.sh"
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

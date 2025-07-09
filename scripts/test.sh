#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🧪 Starte Testdurchlauf..."

python - <<'EOF'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
EOF
HAS_TORCH=$?

if [[ "$HAS_TORCH" -eq 0 ]]; then
    pytest "$@" -q
else
    echo "torch nicht installiert – Heavy-Tests werden übersprungen"
    pytest -m "not heavy" "$@" -q
fi

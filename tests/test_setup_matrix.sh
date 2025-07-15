#!/usr/bin/env bash
set -eu

# Config file missing
rm -f "$HOME/.agentnn_config"
scripts/setup.sh --check >/dev/null || true

# Empty config
: > "$HOME/.agentnn_config"
scripts/setup.sh --check >/dev/null || true

# Invalid value
echo 'POETRY_METHOD="invalid"' > "$HOME/.agentnn_config"
scripts/setup.sh --check >/dev/null || true

# Unset environment variable
unset POETRY_METHOD
scripts/setup.sh --check >/dev/null || true

# Valid default value
echo 'POETRY_METHOD="venv"' > "$HOME/.agentnn_config"

# Run setup with different presets in check mode
scripts/setup.sh --check --full >/dev/null || true
scripts/setup.sh --check --minimal >/dev/null || true
scripts/setup.sh --check --preset dev >/dev/null || true
scripts/setup.sh --check --preset ci >/dev/null || true
scripts/setup.sh --check --preset minimal >/dev/null || true

# Simulate missing tools by clearing PATH
PATH="" scripts/setup.sh --check >/dev/null || true

# Check for interactive prompts
output=$(PATH="" scripts/setup.sh --check 2>&1 || true)
echo "$output" | grep -q "sudo aktivieren" && echo "sudo prompt" || true
echo "$output" | grep -q "installiert werden" && echo "install prompt" || true

# Check for interactive prompts
output=$(PATH="" scripts/setup.sh --check-only 2>&1 || true)
echo "$output" | grep -q "sudo aktivieren" && echo "sudo prompt" || true
echo "$output" | grep -q "installiert werden" && echo "install prompt" || true

# Simulate Poetry install failure with PEP 668
tmpdir=$(mktemp -d)
cat <<'EOF' >"$tmpdir/pip"
#!/bin/bash
echo "error: externally-managed-environment" >&2
exit 1
EOF
chmod +x "$tmpdir/pip"
output=$(printf "2\n" | PATH="$tmpdir" scripts/setup.sh --check 2>&1)
echo "$output" | grep -q "Installation über venv" && echo "poetry menu" || true
echo "$output" | grep -q "Poetry konnte nicht installiert werden" && echo "handled" || true
echo "$output" | grep -q "Zurück zum Hauptmenü" && echo "main menu" || true
grep -q 'POETRY_INSTALL_ATTEMPTED="true"' "$HOME/.agentnn_config" && echo "attempt logged" || true
rm -rf "$tmpdir"

echo "setup matrix executed"

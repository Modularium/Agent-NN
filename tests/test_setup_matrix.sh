#!/usr/bin/env bash
set -eu

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
printf "2\n" | PATH="$tmpdir" scripts/setup.sh --check 2>&1 | grep -q "Installation über venv" && echo "poetry menu" || true
grep -q 'POETRY_INSTALL_METHOD="venv"' .agentnn_config && echo "config stored" || true
rm -rf "$tmpdir"

echo "setup matrix executed"

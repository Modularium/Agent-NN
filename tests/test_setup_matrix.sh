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

echo "setup matrix executed"

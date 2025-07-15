#!/usr/bin/env bash
set -eu

# Run setup with different presets in check mode
scripts/setup.sh --check-only --full >/dev/null || true
scripts/setup.sh --check-only --minimal >/dev/null || true
scripts/setup.sh --check-only --preset dev >/dev/null || true
scripts/setup.sh --check-only --preset ci >/dev/null || true
scripts/setup.sh --check-only --preset minimal >/dev/null || true

# Simulate missing tools by clearing PATH
PATH="" scripts/setup.sh --check-only >/dev/null || true

echo "setup matrix executed"

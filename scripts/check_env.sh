#!/bin/bash
# -*- coding: utf-8 -*-

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers/env.sh"

# Run environment check using the check_environment function
check_environment

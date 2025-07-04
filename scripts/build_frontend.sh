#!/bin/bash
# -*- coding: utf-8 -*-

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers/frontend.sh"

# Build frontend using the build_frontend function
build_frontend

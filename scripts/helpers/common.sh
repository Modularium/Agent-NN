#!/bin/bash
# -*- coding: utf-8 -*-

log_info() { echo -e "\033[1;34m[...]\033[0m $1"; }
log_ok()   { echo -e "\033[1;32m[✓]\033[0m $1"; }
log_warn() { echo -e "\033[1;33m[⚠]\033[0m $1"; }
log_err()  { echo -e "\033[1;31m[✗]\033[0m $1" >&2; }

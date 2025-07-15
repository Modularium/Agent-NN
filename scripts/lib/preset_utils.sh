#!/bin/bash
# -*- coding: utf-8 -*-
# Umfassendes Environment-Repair Script für Agent-NN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
source "$SCRIPT_DIR/lib/status_utils.sh"
source "$SCRIPT_DIR/helpers/common.sh"
source "$SCRIPT_DIR/helpers/env.sh"
source "$SCRIPT_DIR/helpers/docker.sh"

# Repair-Konfiguration
AUTO_FIX=false
DEEP_REPAIR=false
BACKUP_CONFIGS=true
REPAIR_DOCKER=true
REPAIR_PYTHON=true
REPAIR_FRONTEND=true
REPAIR_MCP=true
REPAIR_PERMISSIONS=true

# Repair-Statistiken
declare -A REPAIR_STATS=(
    [attempted]=0
    [successful]=0
    [failed]=0
    [skipped]=0
)

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [COMPONENTS...]

Umfassendes Environment-Repair Script für Agent-NN

Analysiert und repariert automatisch häufige Probleme in der Agent-NN Umgebung.

OPTIONS:
    --auto              Automatische Reparatur ohne Rückfragen
    --deep              Tiefgehende Reparatur (kann länger dauern)
    --no-backup         Keine Backup-Erstellung von Konfigurationsdateien
    --no-docker         Docker-Reparaturen überspringen
    --no-python         Python-Reparaturen überspringen
    --no-frontend       Frontend-Reparaturen überspringen
    --no-mcp            MCP-Reparaturen überspringen
    --no-permissions    Berechtigungs-Reparaturen überspringen
    --dry-run           Nur Probleme anzeigen, nicht reparieren
    -h, --help          Diese Hilfe anzeigen

COMPONENTS:
    system              System-Dependencies und Basis-Tools
    python              Python-Umgebung und Poetry
    docker              Docker und Container-Services
    frontend            Frontend Build und Dependencies
    mcp                 MCP Services und Konfiguration
    permissions         Dateiberechtigungen
    config              Konfigurationsdateien
    
    all                 Alle Komponenten (default)

REPARATUR-KATEGORIEN:
    - Fehlende System-Pakete installieren
    - Python/Poetry Probleme beheben
    - Docker-Berechtigungen reparieren
    - Node.js/npm Issues lösen
    - Konfigurationsdateien wiederherstellen
    - Container und Services neustarten
    - Dateiberechtigungen korrigieren

BEISPIELE:
    $(basename "$0")                    # Interaktive Reparatur
    $(basename "$0") --auto --deep      # Automatische Tiefenreparatur
    $(basename "$0") python docker      # Nur Python und Docker reparieren
    $(basename "$0") --dry-run          # Nur Probleme analysieren

EOF
}

# Logging für Repair-Aktionen
log_repair_action() {
    local action="$1"
    local result="$2"  # attempted|successful|failed|skipped
    local message="$3"
    
    REPAIR_STATS[$result]=$((${REPAIR_STATS[$result]} + 1))
    
    case "$result" in
        attempted)
            log_info "🔧 Repariere: $action"
            ;;
        successful)
            log_ok "✅ Erfolgreich: $action - $message"
            ;;
        failed)
            log_err "❌ Fehlgeschlagen: $action - $message"
            ;;
        skipped)
            log_warn "⏭️ Übersprungen: $action - $message"
            ;;
    esac
}

# Backup-Funktionen
create_backup() {
    local file="$1"
    local backup_dir="$REPO_ROOT/.agentnn/backups/$(date +%Y%m%d-%H%M%S)"
    
    if [[ "$BACKUP_CONFIGS" == "false" ]]; then
        return 0
    fi
    
    if [[ -f "$file" ]]; then
        mkdir -p "$backup_dir"
        local relative_path
        relative_path=$(realpath --relative-to="$REPO_ROOT" "$file")
        local backup_file=

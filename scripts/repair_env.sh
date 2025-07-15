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
        local backup_file="$backup_dir/$relative_path"
        
        mkdir -p "$(dirname "$backup_file")"
        cp "$file" "$backup_file"
        log_debug "Backup erstellt: $backup_file"
    fi
}

# System-Reparaturen
repair_system() {
    log_info "=== SYSTEM-REPARATUR ==="
    
    # Basis-Tools prüfen und installieren
    local required_tools=(curl wget git build-essential)
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool%%-*}" >/dev/null; then  # Entferne Suffixe für check
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_repair_action "Installation fehlender System-Tools: ${missing_tools[*]}" "attempted" ""
        
        if install_packages "${missing_tools[@]}"; then
            log_repair_action "System-Tools Installation" "successful" "${missing_tools[*]} installiert"
        else
            log_repair_action "System-Tools Installation" "failed" "Konnte nicht alle Tools installieren"
        fi
    fi
    
    # Paketmanager-Cache aktualisieren
    log_repair_action "Paketmanager-Cache Update" "attempted" ""
    
    if command -v apt-get >/dev/null; then
        if sudo apt-get update >/dev/null 2>&1; then
            log_repair_action "APT Cache Update" "successful" "Package-Index aktualisiert"
        else
            log_repair_action "APT Cache Update" "failed" "Update fehlgeschlagen"
        fi
    fi
    
    # Disk Space prüfen
    local available_space
    available_space=$(df "$REPO_ROOT" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -lt 2 ]]; then
        log_repair_action "Disk Space Check" "failed" "Nur ${available_gb}GB verfügbar - mindestens 2GB empfohlen"
        
        # Versuche Docker-Cleanup
        if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
            log_repair_action "Docker Cleanup" "attempted" ""
            if docker system prune -f >/dev/null 2>&1; then
                log_repair_action "Docker Cleanup" "successful" "Unbenutzte Docker-Ressourcen entfernt"
            fi
        fi
    fi
}

# Python-Reparaturen
repair_python() {
    log_info "=== PYTHON-REPARATUR ==="
    
    # Python-Installation prüfen
    if ! command -v python3 >/dev/null; then
        log_repair_action "Python3 Installation" "attempted" ""
        
        if ensure_python; then
            log_repair_action "Python3 Installation" "successful" "Python3 installiert"
        else
            log_repair_action "Python3 Installation" "failed" "Python3 konnte nicht installiert werden"
            return 1
        fi
    fi
    
    # python3-venv prüfen und installieren falls nötig
    if ! python3 -m venv --help >/dev/null 2>&1; then
        log_repair_action "python3-venv Installation" "attempted" ""
        
        if install_packages python3-venv; then
            log_repair_action "python3-venv Installation" "successful" "python3-venv installiert"
        else
            log_repair_action "python3-venv Installation" "failed" "python3-venv konnte nicht installiert werden"
        fi
    fi
    
    # pip prüfen
    if ! python3 -m pip --version >/dev/null 2>&1; then
        log_repair_action "pip Installation" "attempted" ""
        
        if python3 -m ensurepip --upgrade >/dev/null 2>&1; then
            log_repair_action "pip Installation" "successful" "pip über ensurepip installiert"
        elif install_packages python3-pip; then
            log_repair_action "pip Installation" "successful" "pip über Paketmanager installiert"
        else
            log_repair_action "pip Installation" "failed" "pip konnte nicht installiert werden"
        fi
    fi
    
    # Poetry prüfen und reparieren
    if ! command -v poetry >/dev/null; then
        log_repair_action "Poetry Installation" "attempted" ""
        
        # Versuche verschiedene Installationsmethoden
        if ensure_poetry_improved; then
            log_repair_action "Poetry Installation" "successful" "Poetry installiert"
        else
            log_repair_action "Poetry Installation" "failed" "Poetry konnte nicht installiert werden"
        fi
    else
        # Poetry ist installiert, prüfe Funktionalität
        cd "$REPO_ROOT" || return 1
        
        if ! poetry check >/dev/null 2>&1; then
            log_repair_action "Poetry Konfiguration Check" "attempted" ""
            
            # Versuche Poetry-Konfiguration zu reparieren
            poetry config virtualenvs.in-project true >/dev/null 2>&1 || true
            
            if poetry check >/dev/null 2>&1; then
                log_repair_action "Poetry Konfiguration" "successful" "Poetry-Konfiguration repariert"
            else
                log_repair_action "Poetry Konfiguration" "failed" "Poetry-Konfiguration konnte nicht repariert werden"
            fi
        fi
        
        # Virtual Environment prüfen
        if [[ ! -d ".venv" ]]; then
            log_repair_action "Python Virtual Environment" "attempted" ""
            
            if poetry install >/dev/null 2>&1; then
                log_repair_action "Python Virtual Environment" "successful" "Dependencies installiert, .venv erstellt"
            else
                log_repair_action "Python Virtual Environment" "failed" "Virtual Environment konnte nicht erstellt werden"
            fi
        fi
    fi
    
    # pyproject.toml Integrität prüfen
    if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        if ! python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null; then
            log_repair_action "pyproject.toml Syntax" "failed" "Syntax-Fehler in pyproject.toml"
        fi
    fi
}

# Docker-Reparaturen
repair_docker() {
    log_info "=== DOCKER-REPARATUR ==="
    
    # Docker-Installation prüfen
    if ! command -v docker >/dev/null; then
        log_repair_action "Docker Installation" "attempted" ""
        
        if ensure_docker; then
            log_repair_action "Docker Installation" "successful" "Docker installiert"
        else
            log_repair_action "Docker Installation" "failed" "Docker konnte nicht installiert werden"
            return 1
        fi
    fi
    
    # Docker-Daemon prüfen
    if ! docker info >/dev/null 2>&1; then
        log_repair_action "Docker Daemon Start" "attempted" ""
        
        # Versuche Docker zu starten
        if sudo systemctl start docker >/dev/null 2>&1; then
            sleep 3
            if docker info >/dev/null 2>&1; then
                log_repair_action "Docker Daemon Start" "successful" "Docker-Daemon gestartet"
            else
                log_repair_action "Docker Daemon Start" "failed" "Docker-Daemon konnte nicht gestartet werden"
            fi
        else
            log_repair_action "Docker Daemon Start" "failed" "systemctl start docker fehlgeschlagen"
        fi
    fi
    
    # Docker-Berechtigungen prüfen
    if ! docker ps >/dev/null 2>&1; then
        log_repair_action "Docker Berechtigungen" "attempted" ""
        
        # Benutzer zur docker-Gruppe hinzufügen
        if sudo usermod -aG docker "$USER" >/dev/null 2>&1; then
            log_repair_action "Docker Berechtigungen" "successful" "Benutzer zur docker-Gruppe hinzugefügt (Neuanmeldung erforderlich)"
        else
            log_repair_action "Docker Berechtigungen" "failed" "Konnte Benutzer nicht zur docker-Gruppe hinzufügen"
        fi
    fi
    
    # Docker Compose prüfen
    if ! docker compose version >/dev/null 2>&1 && ! command -v docker-compose >/dev/null; then
        log_repair_action "Docker Compose Installation" "attempted" ""
        
        # Docker Compose Plugin installieren falls möglich
        if sudo apt-get install -y docker-compose-plugin >/dev/null 2>&1; then
            log_repair_action "Docker Compose Installation" "successful" "Docker Compose Plugin installiert"
        else
            log_repair_action "Docker Compose Installation" "failed" "Docker Compose konnte nicht installiert werden"
        fi
    fi
    
    # Verwaiste Container und Images bereinigen
    if docker info >/dev/null 2>&1; then
        log_repair_action "Docker Cleanup" "attempted" ""
        
        local cleaned_containers
        cleaned_containers=$(docker container prune -f 2>/dev/null | grep "^Deleted" | wc -l || echo "0")
        local cleaned_images
        cleaned_images=$(docker image prune -f 2>/dev/null | grep "^Deleted" | wc -l || echo "0")
        
        log_repair_action "Docker Cleanup" "successful" "$cleaned_containers Container, $cleaned_images Images bereinigt"
    fi
    
    # Compose-Dateien validieren
    local compose_files=("$REPO_ROOT/docker-compose.yml" "$REPO_ROOT/mcp/docker-compose.yml")
    
    for compose_file in "${compose_files[@]}"; do
        if [[ -f "$compose_file" ]]; then
            local file_name
            file_name=$(basename "$(dirname "$compose_file")")/$(basename "$compose_file")
            
            if ! docker compose -f "$compose_file" config >/dev/null 2>&1; then
                log_repair_action "Compose Validation ($file_name)" "failed" "Syntax-Fehler in Compose-Datei"
            fi
        fi
    done
}

# Frontend-Reparaturen
repair_frontend() {
    log_info "=== FRONTEND-REPARATUR ==="
    
    local frontend_dir="$REPO_ROOT/frontend/agent-ui"
    
    if [[ ! -d "$frontend_dir" ]]; then
        log_repair_action "Frontend Verzeichnis" "failed" "Frontend-Verzeichnis nicht gefunden"
        return 1
    fi
    
    cd "$frontend_dir" || return 1
    
    # Node.js prüfen
    if ! command -v node >/dev/null; then
        log_repair_action "Node.js Installation" "attempted" ""
        
        if ensure_node; then
            log_repair_action "Node.js Installation" "successful" "Node.js installiert"
        else
            log_repair_action "Node.js Installation" "failed" "Node.js konnte nicht installiert werden"
            return 1
        fi
    fi
    
    # npm prüfen
    if ! command -v npm >/dev/null; then
        log_repair_action "npm Installation" "attempted" ""
        
        if install_packages npm; then
            log_repair_action "npm Installation" "successful" "npm installiert"
        else
            log_repair_action "npm Installation" "failed" "npm konnte nicht installiert werden"
        fi
    fi
    
    # package.json prüfen
    if [[ ! -f "package.json" ]]; then
        log_repair_action "Frontend package.json" "failed" "package.json nicht gefunden"
        return 1
    fi
    
    # node_modules reparieren
    if [[ ! -d "node_modules" ]] || [[ ! -f "package-lock.json" ]]; then
        log_repair_action "Frontend Dependencies" "attempted" ""
        
        # Bereinige erst alte node_modules
        rm -rf node_modules package-lock.json 2>/dev/null || true
        
        if npm install >/dev/null 2>&1; then
            log_repair_action "Frontend Dependencies" "successful" "npm install erfolgreich"
        else
            log_repair_action "Frontend Dependencies" "failed" "npm install fehlgeschlagen"
        fi
    else
        # Prüfe auf veraltete Dependencies
        if npm outdated >/dev/null 2>&1; then
            log_repair_action "Frontend Dependencies Update" "attempted" ""
            
            if npm update >/dev/null 2>&1; then
                log_repair_action "Frontend Dependencies Update" "successful" "Dependencies aktualisiert"
            else
                log_repair_action "Frontend Dependencies Update" "failed" "Update fehlgeschlagen"
            fi
        fi
    fi
    
    # Build-Verzeichnis prüfen und reparieren
    local dist_dir="$REPO_ROOT/frontend/dist"
    if [[ ! -f "$dist_dir/index.html" ]]; then
        log_repair_action "Frontend Build" "attempted" ""
        
        if npm run build >/dev/null 2>&1; then
            log_repair_action "Frontend Build" "successful" "Frontend neu gebaut"
        else
            log_repair_action "Frontend Build" "failed" "Build fehlgeschlagen"
        fi
    fi
    
    cd "$REPO_ROOT" || return 1
}

# MCP-Reparaturen
repair_mcp() {
    log_info "=== MCP-REPARATUR ==="
    
    local mcp_dir="$REPO_ROOT/mcp"
    
    if [[ ! -d "$mcp_dir" ]]; then
        log_repair_action "MCP Verzeichnis" "failed" "MCP-Verzeichnis nicht gefunden"
        return 1
    fi
    
    # MCP docker-compose.yml prüfen
    if [[ ! -f "$mcp_dir/docker-compose.yml" ]]; then
        log_repair_action "MCP docker-compose.yml" "failed" "MCP docker-compose.yml nicht gefunden"
        return 1
    fi
    
    # MCP .env prüfen
    if [[ ! -f "$mcp_dir/.env" ]]; then
        log_repair_action "MCP Environment" "attempted" ""
        
        if [[ -f "$mcp_dir/.env.example" ]]; then
            create_backup "$mcp_dir/.env.example"
            cp "$mcp_dir/.env.example" "$mcp_dir/.env"
            log_repair_action "MCP Environment" "successful" ".env aus Beispiel erstellt"
        else
            # Minimale .env erstellen
            cat > "$mcp_dir/.env" << 'EOF'
# MCP Services Konfiguration
POSTGRES_DB=mcp_db
POSTGRES_USER=mcp_user
POSTGRES_PASSWORD=mcp_password
REDIS_PASSWORD=mcp_redis_password
MCP_LOG_LEVEL=INFO
MCP_DEBUG=false
EOF
            log_repair_action "MCP Environment" "successful" "Minimale .env erstellt"
        fi
    fi
    
    # MCP Services prüfen und reparieren
    if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
        cd "$mcp_dir" || return 1
        
        # Gestoppte MCP Services neustarten
        local mcp_containers
        mcp_containers=$(docker ps -a --format "{{.Names}}" | grep "mcp-" | wc -l || echo "0")
        
        if [[ $mcp_containers -gt 0 ]]; then
            local running_containers
            running_containers=$(docker ps --format "{{.Names}}" | grep "mcp-" | wc -l || echo "0")
            
            if [[ $running_containers -lt $mcp_containers ]]; then
                log_repair_action "MCP Services Restart" "attempted" ""
                
                if docker_compose up -d >/dev/null 2>&1; then
                    log_repair_action "MCP Services Restart" "successful" "MCP Services neu gestartet"
                else
                    log_repair_action "MCP Services Restart" "failed" "MCP Services konnten nicht gestartet werden"
                fi
            fi
        fi
        
        # MCP Health-Check nach Reparatur
        sleep 5
        local healthy_services=0
        local mcp_ports=(8001 8002 8003 8004 8005)
        
        for port in "${mcp_ports[@]}"; do
            if curl -f -s --max-time 3 "http://localhost:$port/health" >/dev/null 2>&1; then
                healthy_services=$((healthy_services + 1))
            fi
        done
        
        if [[ $healthy_services -gt 0 ]]; then
            log_repair_action "MCP Health Check" "successful" "$healthy_services/${#mcp_ports[@]} Services gesund"
        else
            log_repair_action "MCP Health Check" "failed" "Keine MCP Services antworten"
        fi
        
        cd "$REPO_ROOT" || return 1
    fi
}

# Berechtigungs-Reparaturen
repair_permissions() {
    log_info "=== BERECHTIGUNGS-REPARATUR ==="
    
    # Repository-Berechtigungen
    if [[ ! -w "$REPO_ROOT" ]]; then
        log_repair_action "Repository Schreibrecht" "failed" "Keine Schreibberechtigung für Repository"
    fi
    
    # Script-Ausführungsrechte
    local script_files
    mapfile -t script_files < <(find "$SCRIPT_DIR" -name "*.sh" -type f)
    
    local fixed_scripts=0
    for script in "${script_files[@]}"; do
        if [[ ! -x "$script" ]]; then
            if chmod +x "$script" 2>/dev/null; then
                fixed_scripts=$((fixed_scripts + 1))
            fi
        fi
    done
    
    if [[ $fixed_scripts -gt 0 ]]; then
        log_repair_action "Script Ausführungsrechte" "successful" "$fixed_scripts Scripts repariert"
    fi
    
    # .agentnn Verzeichnis-Berechtigungen
    local agentnn_dir="$REPO_ROOT/.agentnn"
    if [[ -d "$agentnn_dir" ]] && [[ ! -w "$agentnn_dir" ]]; then
        log_repair_action "AgentNN Verzeichnis Berechtigungen" "attempted" ""
        
        if chmod -R u+w "$agentnn_dir" 2>/dev/null; then
            log_repair_action "AgentNN Verzeichnis Berechtigungen" "successful" "Schreibrechte repariert"
        else
            log_repair_action "AgentNN Verzeichnis Berechtigungen" "failed" "Konnte Berechtigungen nicht reparieren"
        fi
    fi
    
    # Log-Verzeichnis erstellen falls nötig
    local log_dir="$REPO_ROOT/logs"
    if [[ ! -d "$log_dir" ]]; then
        log_repair_action "Log Verzeichnis" "attempted" ""
        
        if mkdir -p "$log_dir" 2>/dev/null; then
            log_repair_action "Log Verzeichnis" "successful" "Log-Verzeichnis erstellt"
        else
            log_repair_action "Log Verzeichnis" "failed" "Konnte Log-Verzeichnis nicht erstellen"
        fi
    fi
}

# Konfiguration-Reparaturen
repair_configuration() {
    log_info "=== KONFIGURATIONS-REPARATUR ==="
    
    # .env Hauptdatei
    if [[ ! -f "$REPO_ROOT/.env" ]]; then
        log_repair_action "Haupt .env Datei" "attempted" ""
        
        if [[ -f "$REPO_ROOT/.env.example" ]]; then
            create_backup "$REPO_ROOT/.env.example"
            cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
            log_repair_action "Haupt .env Datei" "successful" ".env aus Beispiel erstellt"
        else
            log_repair_action "Haupt .env Datei" "failed" ".env.example nicht gefunden"
        fi
    fi
    
    # Status-Datei wiederherstellen
    if [[ ! -f "$REPO_ROOT/.agentnn/status.json" ]]; then
        log_repair_action "Status-Datei" "attempted" ""
        
        mkdir -p "$REPO_ROOT/.agentnn"
        echo '{}' > "$REPO_ROOT/.agentnn/status.json"
        log_repair_action "Status-Datei" "successful" "Status-Datei erstellt"
    fi
    
    # Git-Konfiguration prüfen
    if [[ -d "$REPO_ROOT/.git" ]]; then
        if ! git config --get user.name >/dev/null 2>&1; then
            log_repair_action "Git User Config" "attempted" ""
            
            # Setze Standard-Git-Konfiguration
            git config user.name "Agent-NN User" 2>/dev/null || true
            git config user.email "user@agent-nn.local" 2>/dev/null || true
            
            log_repair_action "Git User Config" "successful" "Standard Git-Benutzer gesetzt"
        fi
    fi
}

# Repair-Zusammenfassung
show_repair_summary() {
    echo
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           REPARATUR-ZUSAMMENFASSUNG                         ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    printf "║  Versucht:     %3d Reparaturen                                              ║\n" "${REPAIR_STATS[attempted]}"
    printf "║  ✅ Erfolgreich: %3d Reparaturen                                              ║\n" "${REPAIR_STATS[successful]}"
    printf "║  ❌ Fehlgeschlagen: %3d Reparaturen                                           ║\n" "${REPAIR_STATS[failed]}"
    printf "║  ⏭️ Übersprungen: %3d Reparaturen                                            ║\n" "${REPAIR_STATS[skipped]}"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo
    
    # Empfehlungen
    if [[ ${REPAIR_STATS[failed]} -gt 0 ]]; then
        echo "🔍 WEITERE SCHRITTE:"
        echo "• Führe './scripts/validate.sh --detailed' für detaillierte Diagnose aus"
        echo "• Überprüfe Logs in ./logs/ für weitere Details"
        echo "• Bei Docker-Problemen: Neuanmeldung nach Gruppenmitgliedschaft erforderlich"
        echo "• Bei persistenten Problemen: './scripts/setup.sh --recover' versuchen"
        echo
    fi
    
    if [[ ${REPAIR_STATS[successful]} -gt 0 ]]; then
        echo "✨ NÄCHSTE SCHRITTE:"
        echo "• Führe './scripts/status.sh' aus um den aktuellen Status zu prüfen"
        echo "• Teste die Umgebung mit './scripts/validate.sh'"
        echo "• Starte Services mit './scripts/setup.sh --preset dev'"
        echo
    fi
}

# Hauptfunktion
main() {
    local components=()
    local dry_run=false
    
    # Parameter parsen
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto)
                AUTO_FIX=true
                shift
                ;;
            --deep)
                DEEP_REPAIR=true
                shift
                ;;
            --no-backup)
                BACKUP_CONFIGS=false
                shift
                ;;
            --no-docker)
                REPAIR_DOCKER=false
                shift
                ;;
            --no-python)
                REPAIR_PYTHON=false
                shift
                ;;
            --no-frontend)
                REPAIR_FRONTEND=false
                shift
                ;;
            --no-mcp)
                REPAIR_MCP=false
                shift
                ;;
            --no-permissions)
                REPAIR_PERMISSIONS=false
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            system|python|docker|frontend|mcp|permissions|config)
                components+=("$1")
                shift
                ;;
            all)
                components=(system python docker frontend mcp permissions config)
                shift
                ;;
            *)
                log_err "Unbekannte Option: $1"
                usage >&2
                exit 1
                ;;
        esac
    done
    
    # Standard-Komponenten falls keine angegeben
    if [[ ${#components[@]} -eq 0 ]]; then
        components=(system python docker frontend mcp permissions config)
    fi
    
    log_info "Starte Environment-Reparatur für: ${components[*]}"
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY-RUN Modus: Nur Probleme analysieren"
    fi
    
    if [[ "$AUTO_FIX" == "true" ]]; then
        log_info "AUTO-FIX Modus: Automatische Reparatur aktiviert"
    fi
    
    if [[ "$DEEP_REPAIR" == "true" ]]; then
        log_info "DEEP-REPAIR Modus: Tiefgehende Reparatur aktiviert"
    fi
    
    echo
    
    # Reparaturen ausführen
    for component in "${components[@]}"; do
        case "$component" in
            system)
                [[ "$dry_run" == "false" ]] && repair_system || log_info "SYSTEM: Würde System-Reparaturen durchführen"
                ;;
            python)
                [[ "$REPAIR_PYTHON" == "true" ]] && [[ "$dry_run" == "false" ]] && repair_python || log_info "PYTHON: Würde Python-Reparaturen durchführen"
                ;;
            docker)
                [[ "$REPAIR_DOCKER" == "true" ]] && [[ "$dry_run" == "false" ]] && repair_docker || log_info "DOCKER: Würde Docker-Reparaturen durchführen"
                ;;
            frontend)
                [[ "$REPAIR_FRONTEND" == "true" ]] && [[ "$dry_run" == "false" ]] && repair_frontend || log_info "FRONTEND: Würde Frontend-Reparaturen durchführen"
                ;;
            mcp)
                [[ "$REPAIR_MCP" == "true" ]] && [[ "$dry_run" == "false" ]] && repair_mcp || log_info "MCP: Würde MCP-Reparaturen durchführen"
                ;;
            permissions)
                [[ "$REPAIR_PERMISSIONS" == "true" ]] && [[ "$dry_run" == "false" ]] && repair_permissions || log_info "PERMISSIONS: Würde Berechtigungs-Reparaturen durchführen"
                ;;
            config)
                [[ "$dry_run" == "false" ]] && repair_configuration || log_info "CONFIG: Würde Konfigurations-Reparaturen durchführen"
                ;;
            *)
                log_warn "Unbekannte Komponente: $component"
                ;;
        esac
        echo
    done
    
    # Zusammenfassung anzeigen
    if [[ "$dry_run" == "false" ]]; then
        show_repair_summary
        
        # Exit-Code basierend auf Erfolg
        if [[ ${REPAIR_STATS[failed]} -gt 0 ]]; then
            exit 1
        else
            exit 0
        fi
    else
        log_info "DRY-RUN abgeschlossen - keine Änderungen vorgenommen"
        log_info "Führe Script ohne --dry-run aus um Reparaturen durchzuführen"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

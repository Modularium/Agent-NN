#!/bin/bash
# -*- coding: utf-8 -*-
# Reparatur-Script für bestehende Agent-NN Scripts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Repariere Agent-NN Scripts..."

# 1. Repariere preset_utils.sh
echo "Repariere preset_utils.sh..."
cat > "$SCRIPT_DIR/lib/preset_utils.sh" << 'EOF'
#!/bin/bash
# -*- coding: utf-8 -*-
# Preset definitions used by setup and install scripts.

declare -A PRESETS=(
    [minimal]="Minimale Python-Umgebung"
    [dev]="Vollständige Entwicklungsumgebung"
    [ci]="CI/CD Pipeline Setup"
    [mcp]="MCP Services Setup"
    [full]="Komplettes System mit allen Features"
)

__preset_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
}
__preset_utils_init

validate_preset() {
    local preset="$1"
    if [[ -z "${PRESETS[$preset]+x}" ]]; then
        log_err "Unbekanntes Preset: $preset"
        log_info "Verfügbare Presets:"
        for p in "${!PRESETS[@]}"; do
            log_info "  $p - ${PRESETS[$p]}"
        done
        return 1
    fi
}

apply_preset() {
    local preset="$1"
    validate_preset "$preset" || return 1
    
    PRESET="$preset"
    export PRESET
    
    case "$preset" in
        minimal)
            RUN_MODE="python"
            BUILD_FRONTEND=false
            START_DOCKER=false
            START_MCP=false
            ;;
        dev)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            START_MCP=false
            ;;
        ci)
            RUN_MODE="test"
            BUILD_FRONTEND=false
            START_DOCKER=false
            START_MCP=false
            ;;
        mcp)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            START_MCP=true
            ;;
        full)
            RUN_MODE="full"
            BUILD_FRONTEND=true
            START_DOCKER=true
            START_MCP=true
            ;;
    esac
    
    export RUN_MODE BUILD_FRONTEND START_DOCKER START_MCP
}

export -f validate_preset apply_preset
EOF

# 2. Repariere setup.sh - Poetry-Method Initialisierung
echo "Repariere setup.sh Poetry-Method Initialisierung..."
sed -i '/^POETRY_METHOD=/d' "$SCRIPT_DIR/setup.sh" 2>/dev/null || true

# Füge Poetry-Method Initialisierung nach dem sourcing hinzu
if ! grep -q "POETRY_METHOD.*venv" "$SCRIPT_DIR/setup.sh"; then
    # Suche nach der Zeile mit source preset_utils und füge danach ein
    sed -i '/source.*preset_utils/a\
\
# Poetry-Method initialisieren\
POETRY_METHOD="${POETRY_METHOD:-venv}"\
export POETRY_METHOD' "$SCRIPT_DIR/setup.sh"
fi

# 3. Verbessere install_utils.sh - Poetry Installation
echo "Verbessere install_utils.sh..."
cat >> "$SCRIPT_DIR/lib/install_utils.sh" << 'EOF'

# Verbesserte Poetry-Installation
install_poetry_fixed() {
    log_info "Installiere Poetry mit verbesserter Methode..."
    
    # Prüfe und installiere python3-venv falls nötig
    if ! python3 -m venv --help &>/dev/null; then
        log_info "python3-venv fehlt - installiere System-Paket..."
        require_sudo_if_needed || return 1
        
        if command -v apt-get >/dev/null; then
            $SUDO_CMD apt-get update -y >/dev/null 2>&1
            $SUDO_CMD apt-get install -y python3-venv python3-pip >/dev/null 2>&1
        elif command -v yum >/dev/null; then
            $SUDO_CMD yum install -y python3-venv python3-pip >/dev/null 2>&1
        elif command -v dnf >/dev/null; then
            $SUDO_CMD dnf install -y python3-venv python3-pip >/dev/null 2>&1
        else
            log_err "Kann python3-venv nicht installieren - unbekannter Paketmanager"
            return 1
        fi
    fi
    
    # Versuche Poetry über verschiedene Methoden zu installieren
    case "${POETRY_METHOD:-venv}" in
        system)
            python3 -m pip install --break-system-packages poetry >/dev/null 2>&1
            ;;
        venv)
            if python3 -m venv "$HOME/.agentnn_venv"; then
                source "$HOME/.agentnn_venv/bin/activate"
                pip install poetry >/dev/null 2>&1
                
                # Füge zu .bashrc hinzu falls nicht vorhanden
                if ! grep -q "agentnn_venv" "$HOME/.bashrc" 2>/dev/null; then
                    echo "# Agent-NN Poetry venv" >> "$HOME/.bashrc"
                    echo "source $HOME/.agentnn_venv/bin/activate" >> "$HOME/.bashrc"
                fi
            else
                return 1
            fi
            ;;
        pipx)
            if ! command -v pipx >/dev/null; then
                require_sudo_if_needed || return 1
                $SUDO_CMD apt-get install -y pipx >/dev/null 2>&1
            fi
            pipx install poetry >/dev/null 2>&1
            ;;
    esac
}

# Verbesserte ensure_poetry
ensure_poetry_improved() {
    ensure_pip || return 1
    
    # Prüfe ob Poetry bereits verfügbar ist
    if check_poetry_available; then 
        return 0
    fi
    
    # Im Auto-Modus: Verwende verbesserte Installation
    if [[ "${AUTO_MODE:-false}" == "true" ]]; then
        log_info "Auto-Modus: Installiere Poetry mit verbesserter Methode..."
        POETRY_METHOD="venv"
        export POETRY_METHOD
        save_config_value "POETRY_METHOD" "$POETRY_METHOD"
        install_poetry_fixed || return 1
    else
        # Interaktive Installation
        save_config_value "POETRY_INSTALL_ATTEMPTED" "true"
        install_poetry_interactive || return 130
    fi
    
    if ! check_poetry_available; then
        echo "[✗] Poetry konnte nicht installiert werden."
        return 130
    fi
    return 0
}

# Überschreibe ensure_poetry mit verbesserter Version
ensure_poetry() {
    ensure_poetry_improved "$@"
}
EOF

# 4. Repariere Docker-Helper
echo "Repariere docker.sh..."
# Stelle sicher, dass find_compose_file richtig funktioniert
cat >> "$SCRIPT_DIR/helpers/docker.sh" << 'EOF'

# Repariere find_compose_file falls es Probleme gibt
find_compose_file_fixed() {
    local name="${1:-docker-compose.yml}"
    local prefer_type="${2:-standard}"

    local search_paths=()
    
    case "$prefer_type" in
        mcp)
            search_paths=(
                "$REPO_ROOT/mcp/docker-compose.yml"
                "$REPO_ROOT/mcp/docker-compose.yaml"
                "$REPO_ROOT/docker-compose.mcp.yml"
            )
            ;;
        *)
            search_paths=(
                "$REPO_ROOT/docker-compose.yml"
                "$REPO_ROOT/docker-compose.yaml"
                "$REPO_ROOT/deploy/docker-compose.yml"
            )
            ;;
    esac

    for f in "${search_paths[@]}"; do
        if [ -f "$f" ]; then
            echo "$f"
            return 0
        fi
    done

    log_err "Docker Compose Datei nicht gefunden: $name"
    return 1
}

# Überschreibe find_compose_file falls es existiert
if declare -f find_compose_file >/dev/null; then
    find_compose_file() {
        find_compose_file_fixed "$@"
    }
fi
EOF

# 5. Erstelle einfaches validate.sh falls es fehlt
if [[ ! -f "$SCRIPT_DIR/validate.sh" ]]; then
    echo "Erstelle validate.sh..."
    cat > "$SCRIPT_DIR/validate.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"

log_info "Starte Validierung..."

errors=0

# Basis-Validierung
if ! env_check; then
    errors=$((errors+1))
fi

if ! has_docker || ! docker ps > /dev/null 2>&1; then
    log_err "Docker Container laufen nicht"
    errors=$((errors+1))
fi

if [[ $errors -eq 0 ]]; then
    log_ok "Validierung erfolgreich"
else
    log_warn "Validierung fand $errors Probleme"
fi

exit $errors
EOF
    chmod +x "$SCRIPT_DIR/validate.sh"
fi

# 6. Erstelle einfaches repair_env.sh falls es fehlt
if [[ ! -f "$SCRIPT_DIR/repair_env.sh" ]]; then
    echo "Erstelle repair_env.sh..."
    cat > "$SCRIPT_DIR/repair_env.sh" << 'EOF'
#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"

log_info "Prüfe Umgebung..."
if ! check_environment; then
    log_warn "Versuche fehlende Komponenten zu installieren"
    ensure_python || true
    ensure_poetry_improved || true
    ensure_node || true
    ensure_docker || true
    check_dotenv_file || true
fi
EOF
    chmod +x "$SCRIPT_DIR/repair_env.sh"
fi

# 7. Erstelle einfaches test.sh falls es fehlt
if [[ ! -f "$SCRIPT_DIR/test.sh" ]]; then
    echo "Erstelle test.sh..."
    cp "$SCRIPT_DIR/test_install.sh" "$SCRIPT_DIR/test.sh" 2>/dev/null || {
        cat > "$SCRIPT_DIR/test.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"

echo "🧪 Starte Testdurchlauf..."

if ! command -v python &>/dev/null; then
    echo "Python nicht gefunden. Bitte Python 3.10 oder neuer installieren." >&2
    exit 1
fi

# Basis-Tests
if command -v pytest >/dev/null; then
    pytest -m "not heavy" -q "$@" || {
        echo "❌ Tests fehlgeschlagen" >&2
        return 1
    }
else
    echo "✅ Pytest nicht verfügbar - überspringe Tests"
fi
EOF
    }
    chmod +x "$SCRIPT_DIR/test.sh"
fi

# 8. Setze alle Script-Berechtigungen
echo "Setze Ausführungsberechtigungen..."
chmod +x "$SCRIPT_DIR"/*.sh
chmod +x "$SCRIPT_DIR/agentnn" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/install"/*.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR/deploy"/*.sh 2>/dev/null || true

# 9. Erstelle .agentnn Verzeichnis falls nötig
mkdir -p "$REPO_ROOT/.agentnn"

# 10. Initialisiere POETRY_METHOD in Konfiguration
if [[ ! -f "$REPO_ROOT/.agentnn/config" ]]; then
    echo 'POETRY_METHOD="venv"' > "$REPO_ROOT/.agentnn/config"
fi

echo "✅ Script-Reparatur abgeschlossen!"
echo
echo "Sie können jetzt das Setup ausführen:"
echo "  ./scripts/setup.sh"
echo
echo "Oder das neue CLI-Tool verwenden:"
echo "  ./scripts/agentnn setup"
EOF

chmod +x fix_existing_scripts.sh

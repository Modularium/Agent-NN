#!/bin/bash
# -*- coding: utf-8 -*-

set -euo pipefail

# Only set HELPERS_DIR if it's not already set
if [[ -z "${HELPERS_DIR:-}" ]]; then
    HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Source common.sh from the helpers directory
source "$HELPERS_DIR/common.sh"

readonly FRONTEND_DIR="$HELPERS_DIR/../../frontend/agent-ui"
readonly TARGET_DIST="$HELPERS_DIR/../../frontend/dist"

check_frontend_setup() {
    log_info "Prüfe Frontend-Setup..."
    
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        log_err "Frontend-Verzeichnis nicht gefunden: $FRONTEND_DIR"
        return 1
    fi
    
    if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
        log_err "package.json nicht gefunden in: $FRONTEND_DIR"
        return 1
    fi
    
    log_ok "Frontend-Setup validiert"
    return 0
}

install_frontend_dependencies() {
    log_info "Installiere Frontend-Abhängigkeiten..."
    
    cd "$FRONTEND_DIR" || {
        log_err "Kann nicht in Frontend-Verzeichnis wechseln"
        return 1
    }
    
    # npm ci für reproduzierbare Builds, fallback auf npm install
    if [[ -f "package-lock.json" ]]; then
        if npm ci; then
            log_ok "Frontend-Abhängigkeiten installiert (npm ci)"
        else
            log_warn "npm ci fehlgeschlagen, versuche npm install..."
            if npm install; then
                log_ok "Frontend-Abhängigkeiten installiert (npm install)"
            else
                log_err "Installation der Frontend-Abhängigkeiten fehlgeschlagen"
                return 1
            fi
        fi
    else
        if npm install; then
            log_ok "Frontend-Abhängigkeiten installiert (npm install)"
        else
            log_err "Installation der Frontend-Abhängigkeiten fehlgeschlagen"
            return 1
        fi
    fi
    
    return 0
}

build_frontend() {
    log_info "Baue Frontend..."
    
    if ! check_frontend_setup; then
        return 1
    fi
    
    cd "$FRONTEND_DIR" || {
        log_err "Kann nicht in Frontend-Verzeichnis wechseln"
        return 1
    }
    
    # Abhängigkeiten installieren falls nötig
    if [[ ! -d "node_modules" ]]; then
        if ! install_frontend_dependencies; then
            return 1
        fi
    fi
    
    # Build ausführen
    log_info "Führe Frontend-Build aus..."
    if npm run build; then
        log_ok "Frontend-Build erfolgreich"
    else
        log_err "Frontend-Build fehlgeschlagen"
        return 1
    fi
    
    # Build-Output lokalisieren
    local source_dist=""
    if [[ -d "dist" ]]; then
        source_dist="dist"
    elif [[ -d "../dist" ]]; then
        source_dist="../dist"
    else
        # outDir aus der Vite-Konfiguration ermitteln
        local config_out
        config_out=$(grep -oE "outDir:\s*['\"][^'\"]+['\"]" vite.config.ts | head -n1 | cut -d\" -f2)
        if [[ -n "$config_out" && -d "$config_out" ]]; then
            source_dist="$config_out"
        fi
    fi

    if [[ -z "$source_dist" ]]; then
        log_err "Build-Output nicht gefunden"
        return 1
    fi

    source_dist=$(realpath "$source_dist")
    mkdir -p "$TARGET_DIST"

    if [[ "$source_dist" != "$(realpath "$TARGET_DIST")" ]]; then
        log_info "Kopiere Build-Output von $source_dist nach $TARGET_DIST..."
        if cp -r "$source_dist"/* "$TARGET_DIST"/; then
            log_ok "Build-Output kopiert"
        else
            log_err "Fehler beim Kopieren des Build-Outputs"
            return 1
        fi
    else
        log_debug "Build-Output bereits am Zielort"
    fi

    # Build validieren
    if [[ ! -f "$TARGET_DIST/index.html" ]]; then
        log_err "index.html im Build-Output fehlt: $TARGET_DIST"
        return 1
    fi

    log_ok "Build-Output gefunden: $(realpath "$TARGET_DIST")"

    return 0
}

clean_frontend() {
    log_info "Bereinige Frontend-Build..."
    
    cd "$FRONTEND_DIR" || {
        log_err "Kann nicht in Frontend-Verzeichnis wechseln"
        return 1
    }
    
    # node_modules und dist bereinigen
    local dirs_to_clean=("node_modules" "dist" ".next" ".vite")
    for dir in "${dirs_to_clean[@]}"; do
        if [[ -d "$dir" ]]; then
            log_debug "Entferne $dir..."
            rm -rf "$dir"
        fi
    done
    
    # Target-Dist bereinigen
    if [[ -d "$TARGET_DIST" ]]; then
        log_debug "Entferne $TARGET_DIST..."
        rm -rf "$TARGET_DIST"
    fi
    
    log_ok "Frontend bereinigt"
    return 0
}

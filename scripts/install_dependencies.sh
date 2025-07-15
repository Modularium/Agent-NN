#!/bin/bash
# -*- coding: utf-8 -*-
# Verbesserte System-Abhängigkeiten Installation

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"
source "$SCRIPT_DIR/lib/preset_utils.sh"
source "$SCRIPT_DIR/lib/spinner_utils.sh"

PRESET="dev"
AUTO_INSTALL=false
WITH_SUDO=false

usage() {
    cat <<EOT
Usage: $(basename "$0") [OPTIONS]
Installiere System-Abhängigkeiten für Agent-NN.

Optionen:
  --preset NAME      Vordefiniertes Paketset (dev|ci|minimal|mcp)
  --auto-install     Keine Rückfragen stellen
  --with-sudo        Paketinstallation mit sudo ausführen
  --packages LIST    Spezifische Pakete installieren (kommagetrennt)
  --check-only       Nur prüfen, nicht installieren
  -h, --help         Diese Hilfe anzeigen

Presets:
  dev      - Vollständige Entwicklungsumgebung (Standard)
  ci       - Nur für CI/CD Pipeline benötigte Pakete
  minimal  - Minimale Python-Umgebung
  mcp      - MCP-spezifische Abhängigkeiten

Beispiele:
  $(basename "$0") --preset dev --auto-install
  $(basename "$0") --packages "docker,nodejs,python3"
  $(basename "$0") --check-only
EOT
}

# Erweiterte Paket-Definitionen
declare -A PRESET_PACKAGES=(
    [minimal]="python3 python3-pip python3-venv curl git"
    [ci]="python3 python3-pip python3-venv python3-dev build-essential curl git"
    [dev]="python3 python3-pip python3-venv python3-dev build-essential curl git nodejs npm docker.io docker-compose-plugin"
    [mcp]="python3 python3-pip python3-venv python3-dev build-essential curl git nodejs npm docker.io docker-compose-plugin postgresql-client redis-tools"
)

# Paket-Aliases für verschiedene Distributionen
declare -A PACKAGE_ALIASES=(
    [docker]="docker.io docker-ce"
    [docker-compose]="docker-compose-plugin docker-compose"
    [nodejs]="nodejs node"
    [npm]="npm"
    [python3]="python3"
    [python3-pip]="python3-pip"
    [python3-venv]="python3-venv"
    [python3-dev]="python3-dev python3-devel"
    [build-essential]="build-essential gcc make"
    [postgresql-client]="postgresql-client postgresql"
    [redis-tools]="redis-tools redis"
)

# Erkenne Betriebssystem und Paketmanager
detect_package_manager() {
    if command -v apt-get >/dev/null; then
        echo "apt"
    elif command -v yum >/dev/null; then
        echo "yum"
    elif command -v dnf >/dev/null; then
        echo "dnf"
    elif command -v pacman >/dev/null; then
        echo "pacman"
    elif command -v brew >/dev/null; then
        echo "brew"
    else
        echo "unknown"
    fi
}

# Installiere Paket basierend auf Paketmanager
install_package_by_manager() {
    local package="$1"
    local pkg_manager="$2"
    local sudo_cmd="${3:-}"
    
    log_debug "Installiere $package mit $pkg_manager"
    
    case "$pkg_manager" in
        apt)
            $sudo_cmd apt-get update -y >/dev/null 2>&1 || true
            $sudo_cmd apt-get install -y "$package" >/dev/null 2>&1
            ;;
        yum)
            $sudo_cmd yum install -y "$package" >/dev/null 2>&1
            ;;
        dnf)
            $sudo_cmd dnf install -y "$package" >/dev/null 2>&1
            ;;
        pacman)
            $sudo_cmd pacman -S --noconfirm "$package" >/dev/null 2>&1
            ;;
        brew)
            brew install "$package" >/dev/null 2>&1
            ;;
        *)
            log_err "Unbekannter Paketmanager: $pkg_manager"
            return 1
            ;;
    esac
}

# Prüfe ob Paket installiert ist
is_package_installed() {
    local package="$1"
    local pkg_manager="$2"
    
    case "$pkg_manager" in
        apt)
            dpkg -l "$package" >/dev/null 2>&1
            ;;
        yum|dnf)
            rpm -q "$package" >/dev/null 2>&1
            ;;
        pacman)
            pacman -Q "$package" >/dev/null 2>&1
            ;;
        brew)
            brew list "$package" >/dev/null 2>&1
            ;;
        *)
            # Fallback: prüfe ob Command verfügbar ist
            case "$package" in
                python3) command -v python3 >/dev/null ;;
                nodejs|node) command -v node >/dev/null ;;
                npm) command -v npm >/dev/null ;;
                docker*) command -v docker >/dev/null ;;
                git) command -v git >/dev/null ;;
                curl) command -v curl >/dev/null ;;
                *) return 1 ;;
            esac
            ;;
    esac
}

# Versuche verschiedene Paket-Aliases
try_install_package() {
    local base_package="$1"
    local pkg_manager="$2"
    local sudo_cmd="${3:-}"
    
    # Prüfe erst ob schon installiert
    if is_package_installed "$base_package" "$pkg_manager"; then
        log_debug "$base_package ist bereits installiert"
        return 0
    fi
    
    # Versuche direkte Installation
    if install_package_by_manager "$base_package" "$pkg_manager" "$sudo_cmd"; then
        log_ok "$base_package installiert"
        return 0
    fi
    
    # Versuche Aliases falls definiert
    if [[ -n "${PACKAGE_ALIASES[$base_package]:-}" ]]; then
        local aliases=(${PACKAGE_ALIASES[$base_package]})
        for alias in "${aliases[@]}"; do
            log_debug "Versuche Alias: $alias für $base_package"
            if install_package_by_manager "$alias" "$pkg_manager" "$sudo_cmd"; then
                log_ok "$base_package installiert (als $alias)"
                return 0
            fi
        done
    fi
    
    log_warn "Konnte $base_package nicht installieren"
    return 1
}

# Installiere Liste von Paketen
install_package_list() {
    local packages=("$@")
    local pkg_manager
    local sudo_cmd=""
    local failed_packages=()
    
    pkg_manager=$(detect_package_manager)
    if [[ "$pkg_manager" == "unknown" ]]; then
        log_err "Unbekannter Paketmanager - kann keine Pakete installieren"
        return 1
    fi
    
    log_info "Erkannter Paketmanager: $pkg_manager"
    
    # Sudo-Rechte falls erforderlich
    if [[ "$WITH_SUDO" == "true" ]] || ([[ $(id -u) -ne 0 ]] && [[ "$pkg_manager" != "brew" ]]); then
        if [[ "$AUTO_INSTALL" == "true" ]]; then
            sudo_cmd="sudo"
        else
            local use_sudo
            use_sudo=$(safe_read "Systemrechte (sudo) erforderlich. Verwenden? [J/n]: " 10 "J")
            if [[ "$use_sudo" =~ ^[JjYy]?$ ]]; then
                sudo_cmd="sudo"
            else
                log_err "Installation ohne sudo-Rechte nicht möglich"
                return 1
            fi
        fi
    fi
    
    # Installiere jedes Paket einzeln
    for package in "${packages[@]}"; do
        log_info "Installiere: $package"
        if ! try_install_package "$package" "$pkg_manager" "$sudo_cmd"; then
            failed_packages+=("$package")
        fi
    done
    
    # Berichte über fehlgeschlagene Installationen
    if [[ ${#failed_packages[@]} -gt 0 ]]; then
        log_warn "Folgende Pakete konnten nicht installiert werden:"
        for pkg in "${failed_packages[@]}"; do
            log_warn "  - $pkg"
        done
        return 1
    fi
    
    log_ok "Alle Pakete erfolgreich installiert"
    return 0
}

# Prüfe nur Pakete ohne Installation
check_packages_only() {
    local packages=("$@")
    local pkg_manager
    local missing_packages=()
    
    pkg_manager=$(detect_package_manager)
    
    log_info "Prüfe Paket-Status..."
    
    for package in "${packages[@]}"; do
        if is_package_installed "$package" "$pkg_manager"; then
            log_ok "$package ist installiert"
        else
            log_warn "$package fehlt"
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        echo "Fehlende Pakete: ${missing_packages[*]}"
        return 1
    fi
    
    return 0
}

# Spezielle Docker-Installation
install_docker_improved() {
    local pkg_manager
    pkg_manager=$(detect_package_manager)
    
    log_info "Installiere Docker..."
    
    case "$pkg_manager" in
        apt)
            # Docker's offizieller APT-Repository
            local sudo_cmd=""
            [[ "$WITH_SUDO" == "true" ]] && sudo_cmd="sudo"
            
            $sudo_cmd apt-get update -y >/dev/null
            $sudo_cmd apt-get install -y ca-certificates curl gnupg lsb-release >/dev/null
            
            # Docker GPG Key
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $sudo_cmd gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg 2>/dev/null
            
            # Docker Repository
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | $sudo_cmd tee /etc/apt/sources.list.d/docker.list >/dev/null
            
            $sudo_cmd apt-get update -y >/dev/null
            $sudo_cmd apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin >/dev/null
            ;;
        *)
            # Fallback: Docker's universal installer
            curl -fsSL https://get.docker.com | bash >/dev/null
            ;;
    esac
    
    # Docker service starten
    if command -v systemctl >/dev/null; then
        sudo systemctl enable docker >/dev/null 2>&1 || true
        sudo systemctl start docker >/dev/null 2>&1 || true
    fi
    
    log_ok "Docker installiert"
}

# Hauptfunktion
main() {
    local packages_list=""
    local check_only=false
    
    # Parameter parsen
    while [[ $# -gt 0 ]]; do
        case $1 in
            --preset)
                PRESET="$2"
                shift 2
                ;;
            --auto-install)
                AUTO_INSTALL=true
                AUTO_MODE=true
                shift
                ;;
            --with-sudo)
                WITH_SUDO=true
                shift
                ;;
            --packages)
                packages_list="$2"
                shift 2
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_err "Unbekannte Option: $1"
                usage >&2
                exit 1
                ;;
        esac
    done
    
    # Bestimme zu installierende Pakete
    local packages
    if [[ -n "$packages_list" ]]; then
        IFS=',' read -ra packages <<< "$packages_list"
    else
        if [[ -z "${PRESET_PACKAGES[$PRESET]:-}" ]]; then
            log_err "Unbekanntes Preset: $PRESET"
            exit 1
        fi
        IFS=' ' read -ra packages <<< "${PRESET_PACKAGES[$PRESET]}"
    fi
    
    log_info "Verwende Preset: $PRESET"
    log_info "Zu installierende Pakete: ${packages[*]}"
    
    # Nur prüfen oder installieren
    if [[ "$check_only" == "true" ]]; then
        check_packages_only "${packages[@]}"
    else
        # Spezielle Behandlung für Docker
        local docker_packages=()
        local other_packages=()
        
        for pkg in "${packages[@]}"; do
            if [[ "$pkg" =~ docker ]]; then
                docker_packages+=("$pkg")
            else
                other_packages+=("$pkg")
            fi
        done
        
        # Installiere normale Pakete
        if [[ ${#other_packages[@]} -gt 0 ]]; then
            with_spinner "Installiere System-Pakete" install_package_list "${other_packages[@]}"
        fi
        
        # Installiere Docker speziell
        if [[ ${#docker_packages[@]} -gt 0 ]]; then
            with_spinner "Installiere Docker" install_docker_improved
        fi
        
        log_ok "Installation abgeschlossen"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

#!/bin/bash

__install_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$dir/log_utils.sh"
    source "$dir/../helpers/common.sh"
    source "$dir/spinner_utils.sh"
}

__install_utils_init

: "${POETRY_METHOD:=}"
if [ -z "$POETRY_METHOD" ]; then
    log_warn "Variable POETRY_METHOD war nicht gesetzt – fallback auf venv"
    POETRY_METHOD="venv"
fi

validate_poetry_method() {
    case "$1" in
        system|venv|pipx) return 0 ;;
        *) echo "Ung\xC3\xBCltige POETRY_METHOD: $1"; return 1 ;;
    esac
}

validate_poetry_method "$POETRY_METHOD" || {
    log_warn "Ung\xC3\xBCltige Methode '$POETRY_METHOD' – fallback auf venv"
    POETRY_METHOD="venv"
}

AUTO_MODE="${AUTO_MODE:-false}"
SUDO_CMD="${SUDO_CMD:-}"

# Prompt for sudo if a command requires elevated rights.
require_sudo_if_needed() {
    if [[ $(id -u) -ne 0 && -z "$SUDO_CMD" ]]; then
        log_info "F\xC3\xBCr diesen Schritt werden Systemrechte ben\xC3\xB6tigt."
        read -rp "M\xC3\xB6chtest du 'sudo' aktivieren? [J/n] " ans
        if [[ -z "$ans" || "$ans" =~ ^[JjYy]$ ]]; then
            if ! command -v sudo >/dev/null; then
                read -rp "sudo ist nicht installiert. Jetzt installieren? [J/n] " ans2
                if [[ -z "$ans2" || "$ans2" =~ ^[JjYy]$ ]]; then
                    if command -v apt-get >/dev/null; then
                        apt-get update -y >/dev/null
                        apt-get install -y sudo >/dev/null
                    else
                        log_err "sudo konnte nicht installiert werden"
                        return 1
                    fi
                else
                    return 1
                fi
            fi
            sudo -v || return 1
            SUDO_CMD="sudo"
            log_info "sudo aktiviert"
        else
            return 1
        fi
    fi
    return 0
}

# Ensure a command exists or offer interactive installation.
prompt_and_install_if_missing() {
    local pkg="$1"
    if ! command -v "$pkg" >/dev/null 2>&1; then
        log_warn "$pkg nicht gefunden"
        read -rp "Soll $pkg jetzt installiert werden? [J/n] " ans
        if [[ -z "$ans" || "$ans" =~ ^[JjYy]$ ]]; then
            require_sudo_if_needed || return 1
            install_packages "$pkg" || return 1
            log_ok "Installiert: $pkg"
        else
            log_warn "$pkg nicht installiert"
        fi
    fi
}

require_or_install() {
    local cmd="$1"
    local pkg="${2:-$1}"
    if ! command -v "$cmd" >/dev/null; then
        if ask_install "$pkg"; then
            install_packages "$pkg" || return 1
        else
            return 1
        fi
    fi
    return 0
}

require_or_install_curl() { require_or_install curl curl; }
require_or_install_poetry() { require_or_install poetry poetry; }
require_or_install_nodejs() { require_or_install node nodejs; }
require_or_install_git() { require_or_install git git; }

ensure_git() { require_or_install_git; }
ensure_curl() { require_or_install_curl; }

ask_install() {
    local component="$1"
    if [[ "$AUTO_MODE" == "true" ]]; then
        return 0
    fi
    read -rp "Fehlende Komponente $component gefunden. Jetzt installieren? [J/n] " ans
    if [[ -z "$ans" || "$ans" =~ ^[JjYy]$ ]]; then
        return 0
    fi
    return 1
}

install_docker() {
    log_info "Installiere Docker..."
    require_sudo_if_needed || return 1
    require_or_install_curl || return 1
    curl -fsSL https://get.docker.com | $SUDO_CMD bash >/dev/null
}

ensure_docker() {
    require_or_install_curl || return 1
    if ! command -v docker &>/dev/null; then
        if ask_install "Docker"; then
            install_docker || return 1
        else
            return 1
        fi
    fi
    if ! docker compose version &>/dev/null && ! command -v docker-compose &>/dev/null; then
        log_warn "Docker Compose fehlt"
        # Docker install script includes compose plugin
    fi
    return 0
}

install_node() {
    log_info "Installiere Node.js..."
    require_sudo_if_needed || return 1
    require_or_install_curl || return 1
    curl -fsSL https://deb.nodesource.com/setup_18.x | $SUDO_CMD bash - >/dev/null && $SUDO_CMD apt-get install -y nodejs >/dev/null
}

ensure_node() {
    if ! command -v node &>/dev/null; then
        if ask_install "Node.js"; then
            install_node || return 1
        else
            return 1
        fi
    fi
    if ! command -v npm &>/dev/null; then
        require_sudo_if_needed || return 1
        $SUDO_CMD apt-get install -y npm >/dev/null
    fi
    return 0
}

install_python() {
    log_info "Installiere Python 3.10..."
    require_sudo_if_needed || return 1
    $SUDO_CMD apt-get update -y >/dev/null && \
    $SUDO_CMD apt-get install -y python3.10 python3.10-venv python3.10-distutils >/dev/null
}

ensure_pip() {
    if ! command -v pip >/dev/null; then
        if ask_install "pip"; then
            if command -v apt-get >/dev/null; then
                require_sudo_if_needed || return 1
                $SUDO_CMD apt-get update -y >/dev/null
                $SUDO_CMD apt-get install -y python3-pip >/dev/null && return 0
            fi
            python3 -m ensurepip --upgrade >/dev/null 2>&1 || return 1
        else
            return 1
        fi
    fi
    return 0
}

ensure_python() {
    local version
    if version=$(python3 --version 2>/dev/null); then
        if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)' ; then
            if ask_install "Python 3.10"; then
                install_python || return 1
            else
                return 1
            fi
        fi
    else
        if ask_install "Python 3.10"; then
            install_python || return 1
        else
            return 1
        fi
    fi
}

install_poetry_break_system() {
    log_info "Installiere Poetry systemweit..."
    python3 -m pip install --break-system-packages poetry >/dev/null
}

install_poetry_venv() {
    log_info "Installiere Poetry in ~/.agentnn_venv..."
    python3 -m venv "$HOME/.agentnn_venv" && \
    source "$HOME/.agentnn_venv/bin/activate" && \
    pip install poetry >/dev/null && \
    grep -q "agentnn_venv" "$HOME/.bashrc" 2>/dev/null || echo "source $HOME/.agentnn_venv/bin/activate" >> "$HOME/.bashrc"
}

install_poetry_pipx() {
    log_info "Installiere Poetry via pipx..."
    require_sudo_if_needed || return 1
    $SUDO_CMD apt install pipx -y >/dev/null && pipx install poetry >/dev/null
}

# Try installing Poetry based on chosen method
try_install_poetry() {
    case "$POETRY_METHOD" in
        system) install_poetry_break_system ;;
        venv)   install_poetry_venv ;;
        pipx)   install_poetry_pipx ;;
        *)      install_poetry_interactive ;;
    esac
}

# Check if Poetry is available or in .venv/bin
check_poetry_available() {
    if command -v poetry >/dev/null; then
        return 0
    fi
    local local_poetry="$REPO_ROOT/.venv/bin/poetry"
    if [[ -x "$local_poetry" ]]; then
        echo "→ Lokales Poetry gefunden: .venv/bin/poetry"
        export PATH="$REPO_ROOT/.venv/bin:$PATH"
        return 0
    fi
    return 1
}

map_number_to_method() {
    case "$1" in
        1) echo "system" ;;
        2) echo "venv" ;;
        3) echo "pipx" ;;
    esac
}

prompt_poetry_installation_method() {
    local choice
    while true; do
        cat <<'EOF'
Poetry kann auf deinem System nicht direkt mit pip installiert werden.

[1] Systemweite Installation mit --break-system-packages (nicht empfohlen)
[2] Installation über venv (Standard)
[3] Installation über pipx (empfohlen)
[4] Abbrechen
[q] Zurück zum Hauptmenü
EOF
        read -rp "Bitte wählen [1-4, q]: " choice
        case "$choice" in
            1|2|3)
                POETRY_METHOD="$(map_number_to_method "$choice")"
                export POETRY_METHOD
                save_config_value "POETRY_METHOD" "$POETRY_METHOD"
                break
                ;;
            4)
                echo "🚫 Abgebrochen."
                return 1
                ;;
            q|Q)
                return_to_main_menu
                return 130
                ;;
            *)
                echo "⚠️ Ungültige Eingabe. Bitte 1-4 oder q wählen."
                ;;
        esac
    done
    return 0
}

install_poetry_interactive() {
    local last
    last=$(load_config_value "POETRY_METHOD" "venv")
    if [[ -n "$last" ]]; then
        echo -e "${CYAN}Hinweis:${NC} Du hast bei der letzten Installation '$last' als bevorzugte Methode für Poetry gewählt."
    fi

    if ! prompt_poetry_installation_method; then
        ensure_config_file_exists
        echo -e "→ Schritt übersprungen, keine Installation vorgenommen."
        return 130
    fi

    case "$POETRY_METHOD" in
        system) install_poetry_break_system ;;
        venv)   install_poetry_venv ;;
        pipx)   install_poetry_pipx ;;
    esac || return 1

    echo -e "→ Du hast Option [$POETRY_METHOD] gewählt: Installation über $POETRY_METHOD"
    echo -e "→ Fortsetzung in 3 Sekunden ..."
    sleep 3
    return 0
}

ensure_poetry() {
    ensure_pip || return 1
    check_poetry_available && return 0

    save_config_value "POETRY_INSTALL_ATTEMPTED" "true"

    try_install_poetry || true
    if ! check_poetry_available; then
        echo "[✗] Poetry konnte nicht installiert werden."
        return 130
    fi
    return 0
}

install_python_tool() {
    local tool="$1"
    pip install "$tool" >/dev/null
}

ensure_python_tools() {
    local tools=(ruff mypy pytest)
    for t in "${tools[@]}"; do
        if ! command -v "$t" &>/dev/null; then
            if ask_install "$t"; then
                install_python_tool "$t" || return 1
            fi
        fi
    done
}

export -f ask_install install_docker ensure_docker install_node ensure_node \
          install_python ensure_python ensure_pip install_poetry_interactive \
          prompt_poetry_installation_method map_number_to_method \
          install_poetry_break_system install_poetry_venv install_poetry_pipx \
          try_install_poetry check_poetry_available ensure_poetry \
          install_python_tool ensure_python_tools \
          require_or_install require_or_install_curl \
          require_or_install_poetry require_or_install_nodejs \
          ensure_git ensure_curl require_or_install_git \
          require_sudo_if_needed prompt_and_install_if_missing

# Install a list of system packages using the available package manager.
# Supports apt for Debian/Ubuntu and brew for macOS.
install_packages() {
    local packages=("$@")
    if [[ ${#packages[@]} -eq 0 ]]; then
        return 0
    fi

    local os=""
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        os="$ID"
    else
        os="$(uname -s | tr 'A-Z' 'a-z')"
    fi

    local pkg_list=()
    for pkg in "${packages[@]}"; do
        case "$pkg" in
            poetry)
                if ! command -v poetry >/dev/null; then
                    log_info "Installiere Poetry..."
                    curl -sSL https://install.python-poetry.org | $SUDO_CMD python3 - >/dev/null
                fi
                ;;
            *)
                pkg_list+=("$pkg")
                ;;
        esac
    done

    if [[ ${#pkg_list[@]} -gt 0 ]]; then
        case "$os" in
            ubuntu|debian)
                require_sudo_if_needed || return 1
                $SUDO_CMD apt-get update -y >/dev/null
                $SUDO_CMD apt-get install -y "${pkg_list[@]}" >/dev/null
                ;;
            darwin)
                brew install "${pkg_list[@]}"
                ;;
            *)
                log_err "Betriebssystem $os wird nicht unterstützt"
                return 1
                ;;
        esac
    fi

    return 0
}

export -f install_packages


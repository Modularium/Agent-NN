#!/usr/bin/env sh
# One-click setup: install dependencies, build frontend and start services
set -eu

usage() {
    echo "Usage: $(basename "$0") [--no-services]" >&2
}

NO_SERVICES=false
for arg in "$@"; do
    case $arg in
        --help)
            usage
            exit 0
            ;;
        --no-services)
            NO_SERVICES=true
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

poetry install
scripts/deploy/build_frontend.sh
if ! $NO_SERVICES; then
    scripts/deploy/start_services.sh --build
fi

echo "Setup complete"

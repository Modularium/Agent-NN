#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deploy_docs.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[docs] Prüfe Abhängigkeiten"
if [[ ! -d node_modules ]]; then
  npm ci
fi

echo "[docs] Baue Docusaurus"
npm run build

if [[ ! -d build ]]; then
  echo "[docs] Build fehlgeschlagen" >&2
  exit 1
fi

echo "[docs] Deploye nach gh-pages"
git worktree add -B gh-pages ../gh-pages origin/gh-pages
cp -r build/* ../gh-pages/
git -C ../gh-pages add .
git -C ../gh-pages commit -am "📚 deploy: $(date +%F_%T)"
git -C ../gh-pages push origin gh-pages

echo "[docs] Deployment abgeschlossen"

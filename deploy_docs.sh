#!/bin/bash
set -euo pipefail

# Ensure dependencies
if [[ ! -d node_modules ]]; then
  npm ci
fi

# Build Docusaurus site
npm run build

# Abort if build directory is empty
if [[ -z "$(ls -A build 2>/dev/null)" ]]; then
  echo "Keine Änderungen zum Deployen." >&2
  exit 0
fi

# Mark build for static hosting
touch build/.nojekyll

# Deploy using docusaurus deploy
npm run deploy-docs

echo "Deployment abgeschlossen."

#!/bin/bash
set -euo pipefail

echo "[docs] Checking dependencies"
if [[ ! -d node_modules ]]; then
  npm ci
fi

echo "[docs] Building Docusaurus site"
npm run build

if [[ -z "$(ls -A build 2>/dev/null)" ]]; then
  echo "[docs] Build directory empty, abort" >&2
  exit 1
fi

touch build/.nojekyll

echo "[docs] Deploying to gh-pages branch"
git -C build init
git -C build add .
git -C build commit -m "Deploy docs" || true
git -C build branch -M gh-pages
git -C build remote add origin "$(git config --get remote.origin.url)"
git -C build push -f origin gh-pages

echo "[docs] Deployment abgeschlossen"

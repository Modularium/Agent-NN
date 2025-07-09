#!/bin/bash
set -euo pipefail

echo "[docs] Checking dependencies"
if [[ ! -d node_modules ]]; then
  npm ci
fi

if [[ ! -d build ]]; then
  echo "[docs] Building Docusaurus site"
  npm run build
fi

if [[ ! -d build ]]; then
  echo "[docs] Build directory missing" >&2
  exit 1
fi

touch build/.nojekyll

echo "[docs] Deploying to gh-pages branch"
git -C build init
git -C build add .
timestamp="${CI_DEPLOY_TIMESTAMP:-$(date -u +"%Y-%m-%dT%H:%M:%SZ")}"
git -C build commit -m "Deploy docs $timestamp" || true
git -C build branch -M gh-pages
git -C build remote add origin "$(git config --get remote.origin.url)"
git -C build push -f origin gh-pages

if command -v gh >/dev/null && [[ -n "${GITHUB_REF:-}" ]]; then
  pr_number=${GITHUB_REF##*/}
  gh pr comment "$pr_number" --body "Docs deployed $timestamp" || true
fi

echo "[docs] Deployment abgeschlossen"

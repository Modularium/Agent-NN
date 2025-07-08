#!/bin/bash
set -euo pipefail

BRANCH=gh-pages
REPO=https://github.com/EcoSphereNetwork/Agent-NN.git

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  REPO="https://x-access-token:${GITHUB_TOKEN}@github.com/EcoSphereNetwork/Agent-NN.git"
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "main" ]]; then
  echo "Aktueller Branch ist '$current_branch'. Bitte vorher auf 'main' wechseln." >&2
  exit 1
fi

# Ensure dependencies installed
if [[ ! -d node_modules ]]; then
  npm ci
fi

# Build docs
npm run build

# Check if gh-pages branch exists remotely
if git ls-remote --exit-code --heads origin $BRANCH >/dev/null 2>&1; then
  git fetch origin $BRANCH:$BRANCH
else
  echo "Initialisiere '$BRANCH' Branch..."
  git checkout --orphan $BRANCH
  git reset --hard
  echo "Agent-NN Dokumentation" > README.md
  git add README.md
  git commit -m "Init $BRANCH"
  git push $REPO $BRANCH
  git checkout main
fi

workdir=$(mktemp -d)
trap 'git worktree remove -f "$workdir"; rm -rf "$workdir"' EXIT

# Deploy build output
git worktree add "$workdir" $BRANCH
rm -rf "$workdir"/*
cp -r build/* "$workdir"/
touch "$workdir/.nojekyll"

pushd "$workdir" > /dev/null
if git status --porcelain | grep . >/dev/null; then
  git add .
  git commit -m "Deploy Docusaurus docs"
  git push $REPO $BRANCH
else
  echo "Keine Änderungen zum Deployen."
fi
popd > /dev/null

echo "Deployment abgeschlossen."

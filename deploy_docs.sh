#!/bin/bash
set -e

# ensure we're on main branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "main" ]; then
  echo "Aktueller Branch ist '$current_branch'. Bitte vorher auf 'main' wechseln." >&2
  exit 1
fi

# install dependencies only if node_modules doesn't exist
if [ ! -d node_modules ]; then
  echo "Installiere Node-Abhängigkeiten..."
  npm install
fi

# build docs with docusaurus
npm run build

# check if gh-pages branch exists
if git show-ref --quiet refs/heads/gh-pages; then
  git fetch origin gh-pages:gh-pages
else
  echo "Initialisiere 'gh-pages' Branch..."
  git checkout --orphan gh-pages
  git reset --hard
  echo "Agent-NN Dokumentation" > README.md
  git add README.md
  git commit -m "Init gh-pages"
  git push origin gh-pages
  git checkout main
fi

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

# create worktree for gh-pages
git worktree add "$workdir" gh-pages
rm -rf "$workdir"/*
cp -r build/* "$workdir"/
touch "$workdir/.nojekyll"

cd "$workdir"

git add --all
commit_msg="Deploy documentation"
if git diff --staged --quiet; then
  echo "Keine Änderungen zum Deployen."
else
  git commit -m "$commit_msg"
  git push origin gh-pages
fi

cd -

git worktree remove "$workdir"

echo "gh-pages wurde aktualisiert"
echo "GitHub Pages muss ggf. manuell unter Settings → Pages aktiviert werden (Branch: gh-pages, Ordner: /)"

exit 0

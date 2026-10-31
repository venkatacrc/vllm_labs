#!/usr/bin/env bash
# Rebuilds the course and publishes web/ to the gh-pages branch, which GitHub Pages serves at
# https://venkatacrc.github.io/vllm_labs/. Commits always use the identity below, never the
# global git identity.
# Usage: scripts/publish_web.sh [--no-push]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_ID=(-c user.name="Venkata Chintapalli" -c user.email="venkatc77@gmail.com")
WT="$ROOT/build/gh-pages"
PUSH=1
[ "${1:-}" = "--no-push" ] && PUSH=0

cd "$ROOT"
python3 tools/build_course.py
if [ ! -f web/icons/maskable-512.png ]; then
    swift scripts/make_icon.swift --web web/icons
fi

git worktree remove --force "$WT" 2>/dev/null || true
rm -rf "$WT"
mkdir -p "$(dirname "$WT")"
if git fetch -q origin gh-pages 2>/dev/null; then
    git worktree add -q -B gh-pages "$WT" FETCH_HEAD
elif git show-ref -q --verify refs/heads/gh-pages; then
    git worktree add -q "$WT" gh-pages
else
    git worktree add -q --detach "$WT" HEAD
    git -C "$WT" checkout -q --orphan gh-pages
    git -C "$WT" rm -rfq .
fi

find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
mkdir -p "$WT/icons"
cp web/{index.html,app.css,app.js,course.json,manifest.webmanifest} "$WT/"
cp web/icons/*.png "$WT/icons/"
touch "$WT/.nojekyll"

# A new cache name makes installed copies drop the old files.
VERSION="$(cat "$WT"/*.* web/sw.js | shasum | cut -c1-10)"
sed "s/llm-labs-dev/llm-labs-$VERSION/" web/sw.js > "$WT/sw.js"

git -C "$WT" add -A
if git -C "$WT" diff --cached --quiet; then
    echo "gh-pages already up to date (version $VERSION)"
else
    git -C "$WT" "${GIT_ID[@]}" commit -q -m "Publish web app $VERSION"
    echo "Committed gh-pages (version $VERSION)"
fi
if [ "$PUSH" = 1 ]; then
    git -C "$WT" push -q origin gh-pages
    echo "Pushed. Live at https://venkatacrc.github.io/vllm_labs/ in a minute or two."
fi
git worktree remove --force "$WT"

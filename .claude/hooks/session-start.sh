#!/usr/bin/env bash
# SessionStart hook: catch the two traps every fresh checkout of this repo has —
# a branch cut from a stale/wrong base, and the empty frontend submodule.
# Prints warnings for the session to act on; never blocks session start.
set -u
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

git fetch origin main --quiet 2>/dev/null || true

if git rev-parse --verify --quiet origin/main >/dev/null; then
  base=$(git merge-base HEAD origin/main 2>/dev/null || true)
  if [ -n "$base" ]; then
    behind=$(git rev-list --count "$base..origin/main" 2>/dev/null || echo 0)
    if [ "$behind" -gt 0 ]; then
      echo "WARNING: this branch's merge-base is $behind commit(s) behind origin/main."
      echo "It was likely cut from a stale or wrong base (verify the repo's default branch on GitHub is 'main')."
      echo "Before doing any work: rebase onto origin/main, or if the branch has no unique commits, restart it with 'git checkout -B <branch> origin/main'."
    fi
  fi
fi

if [ -f .gitmodules ] && [ -z "$(ls -A frontend 2>/dev/null)" ]; then
  if git submodule update --init --depth 1 frontend >/dev/null 2>&1; then
    echo "Initialized the frontend/ submodule (it arrives empty in fresh clones)."
  else
    echo "NOTE: frontend/ submodule is empty and could not be initialized (offline?). The web UI will 404; backend endpoints still work."
  fi
fi

exit 0

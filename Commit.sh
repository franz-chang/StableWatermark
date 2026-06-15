#!/usr/bin/env bash
# =============================================================================
# StableWatermark Git Commit & Push Script
# =============================================================================
# 自动提交并推送到 GitHub 仓库
# Usage: ./Commit.sh [commit_message]
# =============================================================================

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT_DIR" ]]; then
  echo "Error: this directory is not inside a git repository."
  exit 1
fi

cd "$ROOT_DIR"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-$(git branch --show-current)}"
DEFAULT_REMOTE_URL="${GIT_REMOTE_URL:-git@github.com:franz-chang/StableWatermark.git}"
RUN_CHECKS="${RUN_CHECKS:-1}"

if [[ -z "$BRANCH" ]]; then
  echo "Error: detached HEAD is not supported by this helper."
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Remote '$REMOTE' was not found. Adding: $DEFAULT_REMOTE_URL"
  git remote add "$REMOTE" "$DEFAULT_REMOTE_URL"
fi

has_remote_branch() {
  git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1
}

branch_ahead_behind() {
  git rev-list --left-right --count HEAD..."$REMOTE/$BRANCH"
}

if [[ "$RUN_CHECKS" == "1" || "$RUN_CHECKS" == "true" ]]; then
  # 检查Python语法 - 排除 venv, lib, .git 等目录
  if command -v python3 >/dev/null 2>&1; then
    echo "Checking Python files..."
    find . \
      -path './.git' -prune -o \
      -path './venv' -prune -o \
      -path './lib' -prune -o \
      -path './site-packages' -prune -o \
      -type f -name '*.py' -print 2>/dev/null | while IFS= read -r pyfile; do
      if ! python3 -m py_compile "$pyfile" 2>/dev/null; then
        echo "  Warning: Syntax error in $pyfile"
      fi
    done
  fi

  # 检查Shell脚本语法 - 同样排除 venv
  echo "Checking shell scripts..."
  find . \
    -path './.git' -prune -o \
    -path './venv' -prune -o \
    -type f -name '*.sh' -print 2>/dev/null | sort | while IFS= read -r script; do
    if ! bash -n "$script" 2>/dev/null; then
      echo "  Warning: Syntax error in $script"
    fi
  done
fi

if [[ "$#" -gt 0 ]]; then
  COMMIT_MESSAGE="$*"
else
  COMMIT_MESSAGE="Update StableWatermark project - $(date '+%Y-%m-%d %H:%M:%S')"
fi

WORKTREE_DIRTY=0
if [[ -n "$(git status --porcelain)" ]]; then
  WORKTREE_DIRTY=1
fi

if [[ "$WORKTREE_DIRTY" -eq 1 ]]; then
  echo "Staging all changes..."
  git add -A

  echo "Committing to $BRANCH..."
  git commit -m "$COMMIT_MESSAGE"
else
  echo "No new working tree changes to commit."
fi

if has_remote_branch; then
  echo "Fetching $REMOTE/$BRANCH..."
  git fetch "$REMOTE" "$BRANCH"

  read -r AHEAD_COUNT BEHIND_COUNT < <(branch_ahead_behind)
  if [[ "$BEHIND_COUNT" -gt 0 ]]; then
    echo "Rebasing local branch onto $REMOTE/$BRANCH..."
    if ! git rebase "$REMOTE/$BRANCH"; then
      echo "Error: rebase failed because of conflicts."
      echo "Resolve conflicts, then run 'git rebase --continue' or cancel with 'git rebase --abort'."
      exit 1
    fi
    read -r AHEAD_COUNT BEHIND_COUNT < <(branch_ahead_behind)
  fi
else
  AHEAD_COUNT="$(git rev-list --count HEAD)"
  BEHIND_COUNT=0
fi

if [[ "$AHEAD_COUNT" -eq 0 && "$BEHIND_COUNT" -eq 0 ]]; then
  echo "No commits to push. Branch is up to date with $REMOTE/$BRANCH."
  exit 0
fi

echo "Pushing to $REMOTE/$BRANCH..."
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push "$REMOTE" "$BRANCH"
else
  git push -u "$REMOTE" "$BRANCH"
fi

echo ""
echo "=============================================="
echo "✅ Done! StableWatermark pushed to GitHub."
echo "=============================================="
echo "Repository: $DEFAULT_REMOTE_URL"
echo "Branch: $BRANCH"
echo "Commit: $COMMIT_MESSAGE"
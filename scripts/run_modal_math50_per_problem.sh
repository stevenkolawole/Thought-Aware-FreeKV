#!/usr/bin/env bash
# Full MATH50 run, one problem per Python subprocess (avoids cross-problem
# CUDA state leak that's been crashing the in-process loop).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal math50_per_problem =====";
  "$MODAL" run --detach scripts/modal_app.py::math50_per_problem;
  rc=$?;
  echo "===== math50_per_problem exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

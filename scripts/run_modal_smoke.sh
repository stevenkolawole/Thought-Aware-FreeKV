#!/usr/bin/env bash
# Smoke test: 1 AIME24 problem, max_gen=512, with instrumentation.
# All stdout/stderr is tee'd to output.log in the repo root.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal smoke =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run scripts/modal_app.py::smoke;
  rc=$?;
  echo "===== smoke exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

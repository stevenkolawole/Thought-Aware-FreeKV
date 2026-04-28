#!/usr/bin/env bash
# MATH50, fetch_interval=2 (fetch every other decode step).
# Writes TBT/timing/recall logs + predictions to tafkv-logs/math50_fetch2/.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$REPO_ROOT/.venv/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal math50_fetch2 =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::math50_fetch2;
  rc=$?;
  echo "===== math50_fetch2 exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

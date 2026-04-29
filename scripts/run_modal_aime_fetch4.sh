#!/usr/bin/env bash
# Single AIME24 problem (2024-I-1), fetch_interval=4 — debug fetch experiment.
# Writes TBT/timing/recall logs to tafkv-logs/aime_debug_fetch4/.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$REPO_ROOT/.venv/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal aime_debug_fetch4 =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::aime_debug_fetch4;
  rc=$?;
  echo "===== aime_debug_fetch4 exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

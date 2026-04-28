#!/usr/bin/env bash
# Run JUST the problem that crashed in the previous math50 run, to verify
# the CUDA event-pool fix. 1h timeout cap.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal math50_test_crash =====";
  "$MODAL" run --detach scripts/modal_app.py::math50_test_crash;
  rc=$?;
  echo "===== math50_test_crash exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

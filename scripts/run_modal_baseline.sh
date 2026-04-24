#!/usr/bin/env bash
# Full FreeKV baseline on AIME24: 29 problems, max_gen=32000, --spec_ret --corr 0.9.
# All stdout/stderr is tee'd to output.log in the repo root.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal baseline =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  # --detach: submit the run to Modal and disconnect the local client.
  # The remote job keeps running to completion even if this laptop sleeps
  # or shuts down. Reattach with `modal app logs <app-id>`.
  "$MODAL" run --detach scripts/modal_app.py::baseline;
  rc=$?;
  echo "===== baseline exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# Re-run 5 specific AIME24 problems with token-level logging to verify that
# cos_sim dips correlate with ThinKV-style transition keywords.
# Detached — survives laptop shutdown. Logs land in the Modal volume under
# `verify_dips/` (separate from the existing `baseline/` data).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal verify_dips =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::verify_dips;
  rc=$?;
  echo "===== verify_dips exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

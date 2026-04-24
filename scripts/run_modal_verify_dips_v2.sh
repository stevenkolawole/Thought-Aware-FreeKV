#!/usr/bin/env bash
# Re-run the original verify_dips 5 AIME24 problems with the enhanced
# logging that caches the full per-head cosine-sim tensor.
# Writes to verify_dips_v2/ subdir on the Modal volume (segregated from
# the earlier verify_dips/ and from dips_v2/).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal verify_dips_v2 =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::verify_dips_v2;
  rc=$?;
  echo "===== verify_dips_v2 exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# Full MATH50 (49 problems) with FreeKV paper-config:
#   max_gen=16384, sink=512, recent=512, corr=0.9, --spec_ret, cuda_cpy backend.
# Per-head cosine sim cached as sims_<pid>.npz. Detached run.
# Writes to volume subdir tafkv-logs/math50/.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal math50 =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::math50;
  rc=$?;
  echo "===== math50 exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# Systems-profiling run on 5 short AIME24 problems.
# Per-component CUDA event timing → timing_<pid>.csv
# Per-step TBT → tbt_<pid>.csv
# Recall log: bytes_actual after need_recall_corr mask
# Writes to volume subdir tafkv-logs/profile_aime/.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal profile_aime =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::profile_aime;
  rc=$?;
  echo "===== profile_aime exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

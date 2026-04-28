#!/usr/bin/env bash
# Full AIME24 (30 problems) with paper-matched config:
#   max_gen=16384 (16K, per Section 5.2 of arXiv 2505.13109)
#   sink=512, recent=512, corr=0.9, --spec_ret, cuda_cpy backend.
# Detached run; survives laptop shutdown. Logs land in tafkv-logs/full_aime/.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal full_aime =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::full_aime;
  rc=$?;
  echo "===== full_aime exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# Run 5 NEW AIME24 problems (not in baseline/ or verify_dips/) with the
# enhanced logging system that caches per-(step, layer, q_head) cosine sim
# as a compressed npz sidecar. Detached so laptop shutdown is fine.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
LOG_FILE="$REPO_ROOT/output.log"

cd "$REPO_ROOT"
{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) modal dips_v2 =====";
  echo "modal:  $MODAL";
  echo "app:    scripts/modal_app.py";
  "$MODAL" run --detach scripts/modal_app.py::dips_v2;
  rc=$?;
  echo "===== dips_v2 exit=$rc =====";
  exit $rc;
} 2>&1 | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# Download instrumentation CSVs from the Modal logs volume.
# Usage: ./scripts/fetch_logs.sh [subdir]   (default: everything)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODAL="${MODAL:-$HOME/.venvs/main/bin/modal}"
VOL="tafkv-logs"
SUB="${1:-}"
DEST="$REPO_ROOT/modal_logs${SUB:+/$SUB}"

mkdir -p "$DEST"
echo "[fetch_logs] $VOL/$SUB -> $DEST"
if [[ -n "$SUB" ]]; then
  "$MODAL" volume get "$VOL" "$SUB" "$DEST"
else
  "$MODAL" volume get "$VOL" / "$DEST"
fi

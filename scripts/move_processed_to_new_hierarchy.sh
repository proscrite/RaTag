#!/usr/bin/env bash
set -euo pipefail

# Move processed_data folders into a central processed/<run_id> hierarchy.
# Usage:
#   ./move_processed_to_new_hierarchy.sh [--dry-run] [--data-root /path/to/data_root] [run_dir]
# Examples:
#   ./move_processed_to_new_hierarchy.sh --dry-run
#   ./move_processed_to_new_hierarchy.sh /Volumes/KINGSTON/RaTag_data/RUN22_Th228 --dry-run

DRY_RUN=true
DATA_ROOT_DEFAULT="/Volumes/KINGSTON/RaTag_data"

show_help(){
  sed -n '1,120p' "$0" | sed -n '1,12p'
}

POSITIONAL=()
DATA_ROOT="${DATA_ROOT_DEFAULT}"
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run|-n)
      DRY_RUN=true; shift ;;
    --apply|-a)
      DRY_RUN=false; shift ;;
    --data-root)
      DATA_ROOT="$2"; shift 2 ;;
    --help|-h)
      show_help; exit 0 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done

if [ ${#POSITIONAL[@]} -gt 0 ]; then
  RUN_PATH="${POSITIONAL[0]}"
  if [ ! -d "$RUN_PATH" ]; then
    echo "Run path not found: $RUN_PATH"; exit 1
  fi
  RUNS=("$RUN_PATH")
else
  # Prefer explicit RUN* folders directly under data root (e.g. /Volumes/.../RUN22_Th228)
  RUNS=()
  CANDIDATES=()
  while IFS= read -r -d '' d; do
    CANDIDATES+=("$d")
  done < <(find "$DATA_ROOT" -maxdepth 1 -type d -name "RUN*" -print0 2>/dev/null || true)
  for c in "${CANDIDATES[@]}"; do
    if [ -d "$c/processed_data" ]; then
      RUNS+=("$c")
    fi
  done

  # Fallback: if no RUN* folders with processed_data, search deeper for processed_data dirs
  if [ ${#RUNS[@]} -eq 0 ]; then
    RUNS=()
    while IFS= read -r -d '' d; do
      RUNS+=("$(dirname "$d")")
    done < <(find "$DATA_ROOT" -maxdepth 2 -type d -name processed_data -print0 2>/dev/null || true)
  fi
fi

if [ ${#RUNS[@]} -eq 0 ]; then
  echo "No runs with processed_data found under $DATA_ROOT"
  exit 0
fi

echo "Found ${#RUNS[@]} runs to process"

for R in "${RUNS[@]}"; do
  RUN_ID=$(basename "$R")
  SRC="$R/processed_data/"
  # Determine processed root from env if set, else parent of data root + /processed
  if [ -n "${RATAG_PROCESSED_ROOT-}" ]; then
    DEST_ROOT="$RATAG_PROCESSED_ROOT"
  else
    # assume R is like /.../RUN22_Th228
    DATA_ROOT_PARENTS=$(dirname "$R")
    # pick the parent directory of the run folder
    DEST_ROOT="${DATA_ROOT_PARENTS%/}/processed"
  fi
  DEST="$DEST_ROOT/$RUN_ID/"

  mkdir -p "$DEST"

  if [ "$DRY_RUN" = true ]; then
    echo "DRY-RUN: rsync -avn --progress '$SRC' '$DEST'"
    rsync -avn --progress "$SRC" "$DEST"
  else
    echo "Copying $SRC -> $DEST (this may take a while)"
    rsync -av --checksum --progress "$SRC" "$DEST"
    echo "Verifying (checksum)"
    rsync -avnc --progress "$SRC" "$DEST"
    echo "Move completed for $RUN_ID"
  fi

  # Also move energy_maps if present
  if [ -d "$R/energy_maps" ]; then
    if [ "$DRY_RUN" = true ]; then
      echo "DRY-RUN: rsync -avn --progress '$R/energy_maps/' '$DEST/energy_maps/'"
      rsync -avn --progress "$R/energy_maps/" "$DEST/energy_maps/"
    else
      mkdir -p "$DEST/energy_maps"
      rsync -av --checksum --progress "$R/energy_maps/" "$DEST/energy_maps/"
      echo "Moved energy_maps for $RUN_ID"
    fi
  fi

  # Also move plots; copy both plots/ and plots/all/ contents into DEST/plots/
  if [ -d "$R/plots" ]; then
    if [ "$DRY_RUN" = true ]; then
      echo "DRY-RUN: rsync -avn --progress '$R/plots/' '$DEST/plots/'"
      rsync -avn --progress "$R/plots/" "$DEST/plots/" || true
      if [ -d "$R/plots/all" ]; then
        echo "DRY-RUN: rsync -avn --progress '$R/plots/all/' '$DEST/plots/'"
        rsync -avn --progress "$R/plots/all/" "$DEST/plots/" || true
      fi
    else
      mkdir -p "$DEST/plots"
      rsync -av --checksum --progress "$R/plots/" "$DEST/plots/" || true
      if [ -d "$R/plots/all" ]; then
        rsync -av --checksum --progress "$R/plots/all/" "$DEST/plots/" || true
      fi
      echo "Moved plots for $RUN_ID"
    fi
  fi
done

echo "All done. If you ran without --dry-run, verify results before removing original processed_data directories."

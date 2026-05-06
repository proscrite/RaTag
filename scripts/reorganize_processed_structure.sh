#!/usr/bin/env bash
set -euo pipefail

# Reorganize processed/<run_id>/ directories into a cleaner hierarchy.
# Moves:
#  - *_t_s1.npz -> t_s1/
#  - *_t_s2.npz -> t_s2/
#  - *_s2_areas.npz -> s2_areas/
#  - *_xray_areas.npz -> s2_areas/
#  - *_metadata.json -> set_summaries/
#  - RUN*_recomb_*.csv, RUN*_recombination_plots.png, RUN*_s2_vs_drift.csv -> run_summaries/
# Usage:
#   ./scripts/reorganize_processed_structure.sh [--dry-run] [--processed-root /path/to/processed]
# Examples:
#   ./scripts/reorganize_processed_structure.sh --dry-run
#   ./scripts/reorganize_processed_structure.sh --apply --processed-root /Volumes/KINGSTON/RaTag_data/processed

DRY_RUN=true
PROCESSED_ROOT_DEFAULT="/Volumes/KINGSTON/RaTag_data/processed"
PROCESSED_ROOT="${PROCESSED_ROOT_DEFAULT}"

usage(){
  sed -n '1,120p' "$0" | sed -n '1,20p'
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run|-n)
      DRY_RUN=true; shift ;;
    --apply|-a)
      DRY_RUN=false; shift ;;
    --processed-root)
      PROCESSED_ROOT="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [ ! -d "$PROCESSED_ROOT" ]; then
  echo "Processed root not found: $PROCESSED_ROOT"; exit 1
fi

echo "Processed root: $PROCESSED_ROOT"
echo "Mode: $( [ "$DRY_RUN" = true ] && echo dry-run || echo apply )"

unique_dest(){
  local dest="$1"
  if [ ! -e "$dest" ]; then
    printf '%s' "$dest"
    return
  fi
  local base ext i
  base="${dest%.*}"
  ext="${dest##*.}"
  i=1
  while [ -e "${base}.${i}.${ext}" ]; do
    i=$((i+1))
  done
  printf '%s' "${base}.${i}.${ext}"
}

process_run(){
  local run_dir="$1"
  echo "\nProcessing run: $run_dir"

  mkdir_cmds=(
    "$run_dir/t_s1"
    "$run_dir/t_s2"
    "$run_dir/s2_areas"
    "$run_dir/set_summaries"
    "$run_dir/run_summaries"
  )
  for d in "${mkdir_cmds[@]}"; do
    if [ "$DRY_RUN" = true ]; then
      echo "DRY-RUN: mkdir -p '$d'"
    else
      mkdir -p "$d"
    fi
  done

  # patterns -> destination subdir (use parallel arrays for macOS bash compatibility)
  PATTERNS=(
    "*_t_s1.npz"
    "*_t_s2.npz"
    "*_s2_areas.npz"
    "*_xray_areas.npz"
    "*_metadata.json"
    "RUN*_recomb_*.csv"
    "RUN*_recombination_plots*.png"
    "RUN*_s2_vs_drift*.csv"
  )
  DESTS=(
    t_s1
    t_s2
    s2_areas
    s2_areas
    set_summaries
    run_summaries
    run_summaries
    run_summaries
  )

  for idx in "${!PATTERNS[@]}"; do
    pat=${PATTERNS[$idx]}
    dest_sub=${DESTS[$idx]}
    # find matching files at depth 1 or 2 (to catch files in all/)
    while IFS= read -r -d '' f; do
      fname=$(basename "$f")
      dest="$run_dir/$dest_sub/$fname"
      if [ "$DRY_RUN" = true ]; then
        echo "DRY-RUN: mv '$f' '$dest'"
      else
        final_dest=$(unique_dest "$dest")
        mv "$f" "$final_dest"
        echo "Moved: $f -> $final_dest"
      fi
    done < <(find "$run_dir" -maxdepth 2 -type f -name "$pat" -print0 2>/dev/null || true)
  done
}

# Iterate runs
runs=()
while IFS= read -r -d '' d; do
  runs+=("$d")
done < <(find "$PROCESSED_ROOT" -maxdepth 1 -type d -name "RUN*" -print0 2>/dev/null || true)


# Also scan parent data root for run dirs that still have processed_data or 'all' folders
DATA_ROOT_PARENT=$(dirname "$PROCESSED_ROOT")
extra_runs=()
while IFS= read -r -d '' rr; do
  # rr is something like /Volumes/.../RUN26_...
  if [ -d "$rr/processed_data" ] || [ -d "$rr/all" ]; then
    # if not already in runs, add
    skip=false
    for existing in "${runs[@]}"; do
      if [ "$existing" = "$rr" ]; then skip=true; break; fi
    done
    if [ "$skip" = false ]; then
      extra_runs+=("$rr")
    fi
  fi
done < <(find "$DATA_ROOT_PARENT" -maxdepth 1 -type d -name "RUN*" -print0 2>/dev/null || true)

if [ ${#runs[@]} -eq 0 ] && [ ${#extra_runs[@]} -eq 0 ]; then
  echo "No RUN* directories found under $PROCESSED_ROOT or $DATA_ROOT_PARENT"; exit 0
fi

for r in "${runs[@]}"; do
  process_run "$r"
done

# Process runs that still live in the original data root (move their processed files into PROCESSED_ROOT)
for rr in "${extra_runs[@]}"; do
  run_id=$(basename "$rr")
  echo "\nProcessing original-run files for: $rr -> will move into $PROCESSED_ROOT/$run_id"
  dest_run_dir="$PROCESSED_ROOT/$run_id"
  if [ "$DRY_RUN" = true ]; then
    echo "DRY-RUN: mkdir -p '$dest_run_dir'"
  else
    mkdir -p "$dest_run_dir"
  fi

  # For each pattern, move files from rr/processed_data or rr/all into the dest_run_dir subfolders
  src_candidates=("$rr/processed_data" "$rr")
  for src_base in "${src_candidates[@]}"; do
    if [ ! -d "$src_base" ]; then
      continue
    fi
    for idx in "${!PATTERNS[@]}"; do
      pat=${PATTERNS[$idx]}
      dest_sub=${DESTS[$idx]}
      while IFS= read -r -d '' f; do
        fname=$(basename "$f")
        dest="$dest_run_dir/$dest_sub/$fname"
        if [ "$DRY_RUN" = true ]; then
          echo "DRY-RUN: mv '$f' '$dest'"
        else
          mkdir -p "$dest_run_dir/$dest_sub"
          final_dest=$(unique_dest "$dest")
          mv "$f" "$final_dest"
          echo "Moved: $f -> $final_dest"
        fi
      done < <(find "$src_base" -maxdepth 2 -type f -name "$pat" -print0 2>/dev/null || true)
    done
  done
done

echo "\nReorganization complete (dry-run=$DRY_RUN). Review results before deleting originals."

#!/usr/bin/env bash
set -euo pipefail

# Optional helper: run after each GENPEPT output exists.
# Expects compare_genpept_exploration.py in current directory or set COMPARE_PY.
COMPARE_PY=${COMPARE_PY:-compare_genpept_exploration.py}

outputs=(
  chignolin_genpept_run01_full_heavy_max
  chignolin_genpept_run02_raw_generation_only
  chignolin_genpept_run03_bh_only
  chignolin_genpept_run04_nma_only
  chignolin_genpept_run05_minimal_full
  chignolin_genpept_run06_pca_frontier_control
  chignolin_genpept_run07_reasonable_all_methods
)

for out in "${outputs[@]}"; do
  if [[ -d "${out}" ]]; then
    echo "=== Comparing ${out} ==="
    python "${COMPARE_PY}" "${out}" \
      --auto-final-report \
      --write-json "${out}/exploration_comparison.json" \
      --write-stage-csv "${out}/exploration_stage_deltas.csv" \
      --write-timing-csv "${out}/timing_summary.csv" --write-figures "${out}"
  else
    echo "Skipping ${out}: directory does not exist"
  fi
done

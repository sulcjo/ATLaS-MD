#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing GENPEPT.py.
# Edit GENPEPT_PY if your script lives elsewhere.
GENPEPT_PY=${GENPEPT_PY:-../GENPEPT.py}

configs=(
  genpept_run01_full_heavy_max.yaml
  genpept_run02_raw_generation_control.yaml
  genpept_run03_bh_only_control.yaml
  genpept_run04_nma_only_control.yaml
  genpept_run05_minimal_full_necessary.yaml
  genpept_run06_pca_frontier_control.yaml
  genpept_run07_reasonable_all_methods.yaml
)

for cfg in "${configs[@]}"; do
  echo "=== Running ${cfg} ==="
  python "${GENPEPT_PY}" --config "${cfg}"
done

#!/usr/bin/env bash
# Full Ace-Ala-Nme phi/psi validation: reference -> validation -> 3x analysis -> compare.
# Run from repo root. Outputs under RUNS/ala_dipeptide_validation/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"

ROOT=RUNS/ala_dipeptide_validation
LOG=$ROOT/logs
mkdir -p "$LOG"

echo "[1/5] fixture"
python ala_dipeptide_fixture.py "$ROOT/ace_ala_nme.pdb"

if [ -f "$ROOT/reference/replica_trajectories/replica_000.xtc" ]; then
  echo "[2/5] reference already present -> skipping"
else
  echo "[2/5] reference (unbiased, 100 ns) -> $LOG/reference.log"
  python -m gareus --config ala_dipeptide_reference.yaml > "$LOG/reference.log" 2>&1
fi

echo "[3/5] validation (hmr-gamd/REUS 2D, 20x5 ns) -> $LOG/validation.log"
python -m gareus --config ala_dipeptide_validation.yaml > "$LOG/validation.log" 2>&1

# Solute-only trajectories => analysis must read solute_only.pdb as topology.
SOLUTE_TOP="$ROOT/validation/solute_only.pdb"
echo "[4/5] analysis (gamd_cumulant2 only) using $SOLUTE_TOP"
python analyze_gareus_mbar.py "$ROOT/validation" \
  --selected-method gamd_cumulant2 \
  --extra-topology "$SOLUTE_TOP" --rg-topology "$SOLUTE_TOP" --pca-topology "$SOLUTE_TOP" \
  --rg-selection "all" --pca-selection "name CA" \
  --out "$ROOT/validation/ANALYSIS_gamd_cumulant2" > "$LOG/analysis_gamd_cumulant2.log" 2>&1

echo "[5/5] validate (reference vs gamd_cumulant2)"
python validate_ala_dipeptide.py \
  --reference-run "$ROOT/reference" \
  --rama-npz \
    "$ROOT"/validation/ANALYSIS_gamd_cumulant2/extra_observable_pmfs/ramachandran_2d_fes/rama_*_2d_fes.npz \
  --estimator-names gamd_cumulant2 \
  --temperature-k 300.0 \
  --out "$ROOT/VALIDATION_REPORT" | tee "$LOG/validate.log"

echo "DONE. Report: $ROOT/VALIDATION_REPORT/validation_report.json"

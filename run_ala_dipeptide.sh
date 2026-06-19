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

echo "[2/5] reference (unbiased, 100 ns) -> $LOG/reference.log"
python -m gareus --config ala_dipeptide_reference.yaml > "$LOG/reference.log" 2>&1

echo "[3/5] validation (hmr-gamd/REUS 2D, 20x5 ns) -> $LOG/validation.log"
python -m gareus --config ala_dipeptide_validation.yaml > "$LOG/validation.log" 2>&1

echo "[4/5] analysis x3 (one per estimator)"
for sel in umbrella_only gamd_exponential gamd_cumulant2; do
  echo "    --selected-method $sel"
  python analyze_gareus_mbar.py "$ROOT/validation" \
    --selected-method "$sel" \
    --out "$ROOT/validation/ANALYSIS_$sel" > "$LOG/analysis_$sel.log" 2>&1
done

echo "[5/5] validate (reference vs 3 estimators)"
python validate_ala_dipeptide.py \
  --reference-run "$ROOT/reference" \
  --rama-npz \
    "$ROOT"/validation/ANALYSIS_umbrella_only/extra_observable_pmfs/ramachandran_2d_fes/rama_*_2d_fes.npz \
    "$ROOT"/validation/ANALYSIS_gamd_exponential/extra_observable_pmfs/ramachandran_2d_fes/rama_*_2d_fes.npz \
    "$ROOT"/validation/ANALYSIS_gamd_cumulant2/extra_observable_pmfs/ramachandran_2d_fes/rama_*_2d_fes.npz \
  --estimator-names umbrella_only gamd_exponential gamd_cumulant2 \
  --temperature-k 300.0 \
  --out "$ROOT/VALIDATION_REPORT" | tee "$LOG/validate.log"

echo "DONE. Report: $ROOT/VALIDATION_REPORT/validation_report.json"

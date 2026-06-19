#!/usr/bin/env bash
# Ace-Ala-Nme phi/psi validation: reference -> validation -> cumulant2 analysis -> compare.
# Run from repo root. Outputs under RUNS/ala_dipeptide_validation/ (gitignored, so on a
# fresh cluster checkout everything here is regenerated from scratch).
#
# RESUME-AWARE: safe to resubmit the same sbatch repeatedly under a short wall-clock.
# Each MD stage is, per invocation:
#   * complete (prod_done >= production_steps)       -> skipped
#   * started  (checkpoint manifest present)         -> --resume (continues)
#   * not started                                    -> fresh (sets up + starts production)
# IMPORTANT: the FIRST job must run WITHOUT --resume (manual-mode --resume errors when no
# checkpoint exists yet) -- this script handles that automatically; do not pass --resume.
# NOTE: final_report.md is written on graceful shutdown too, so it is NOT a completion
# marker; completion is decided from the checkpoint manifest's prod_done vs production_steps.
# No `set -e`: a SLURM-killed stage just leaves a checkpoint for the next resubmit.

cd "$(dirname "$0")"

ROOT=RUNS/ala_dipeptide_validation
LOG=$ROOT/logs
mkdir -p "$LOG"

# Path to a stage's production checkpoint manifest (top-level or final_production).
manifest_path() {
  local rundir="$1"
  if [ -f "$rundir/checkpoints/production_checkpoint_manifest.json" ]; then
    echo "$rundir/checkpoints/production_checkpoint_manifest.json"
  elif [ -f "$rundir/final_production/checkpoints/production_checkpoint_manifest.json" ]; then
    echo "$rundir/final_production/checkpoints/production_checkpoint_manifest.json"
  fi
}

# Return 0 if prod_done >= production_steps (run_args.json), else 1.
stage_complete() {
  local rundir="$1" mf
  mf="$(manifest_path "$rundir")"
  [ -n "$mf" ] || return 1
  python3 - "$mf" "$rundir/run_args.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
try:
    a = json.load(open(sys.argv[2]))
except Exception:
    a = {}
done = int(m.get("prod_done", 0))
target = int(a.get("production_steps", 0) or 0)
sys.exit(0 if target > 0 and done >= target else 1)
PY
}

# Run one gareus stage, resuming/skipping based on on-disk state.
# args: <config.yaml> <run_dir> <label> <logfile>
run_stage() {
  local config="$1" rundir="$2" label="$3" logf="$4"
  if stage_complete "$rundir"; then
    echo "[$label] complete (prod_done >= production_steps) -> skip"
    return 0
  fi
  local resume=""
  if [ -n "$(manifest_path "$rundir")" ]; then
    resume="--resume"
    echo "[$label] checkpoint found -> --resume -> $logf"
  else
    echo "[$label] fresh start -> $logf"
  fi
  python -m gareus --config "$config" $resume > "$logf" 2>&1 || true
  if stage_complete "$rundir"; then
    echo "[$label] completed this job"
  else
    echo "[$label] not finished (killed/error/partial) -> resubmit to continue; see $logf"
  fi
}

echo "[1/5] fixture"
python ala_dipeptide_fixture.py "$ROOT/ace_ala_nme.pdb"

echo "[2/5] reference (unbiased, 100 ns)"
run_stage ala_dipeptide_reference.yaml "$ROOT/reference" "reference" "$LOG/reference.log"

# Validation only makes sense once the reference is fully sampled; else resubmit.
if ! stage_complete "$ROOT/reference"; then
  echo "Reference not complete yet -> stopping here. Resubmit to continue the reference."
  exit 0
fi

echo "[3/5] validation (hmr-gamd/REUS, dense gamd_cumulant2)"
run_stage ala_dipeptide_validation.yaml "$ROOT/validation" "validation" "$LOG/validation.log"

if ! stage_complete "$ROOT/validation"; then
  echo "Validation not complete yet -> stopping before analysis. Resubmit to continue."
  exit 0
fi

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

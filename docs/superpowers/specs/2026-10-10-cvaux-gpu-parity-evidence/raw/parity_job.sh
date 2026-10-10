#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
# Usage: sbatch --job-name=cvp_<MODE> parity_job.sh <MODE>
# MODE: prep | cuda_mixed | opencl | cuda_double | roundtrip | noaux
# CVaux GPU parity runbook (docs/superpowers/specs/2026-10-07-cvaux-gpu-parity-runbook.md).
# Scratch CODE_DIR only; never touches ~/2026_peptide_sampler or ~/gareus.
set -Eeuo pipefail
MODE="$1"
set --   # activate scripts read $1 as an env name
CODE_DIR=/home/sulcjo/2026_peptide_sampler_cvaux_parity
SCR=/home/sulcjo/cvaux_parity_runs
cd "${CODE_DIR}"

source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc
export OPENMM_CPU_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
unset PYTHONPATH; export PYTHONPATH="${CODE_DIR}"
export CUDA_CACHE_PATH="${TMPDIR:-/tmp}/cuda_kernel_cache_${SLURM_JOB_ID:-$$}"; mkdir -p "${CUDA_CACHE_PATH}"
ulimit -n 65536 || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
echo "[cvp] mode=${MODE} commit=$(cat ${CODE_DIR}/DEPLOYED_COMMIT) host=$(hostname) job=${SLURM_JOB_ID:-none} CVD=${CUDA_VISIBLE_DEVICES:-}"

F="--seq GA --cv1 distance --window-mode manual --exchange-mode gibbs-walk --seed 7 --minimize-iterations 200 \
   --npt-steps 2000 --us-pull-steps-per-window 500 --production-steps 6000 --checkpoint-interval 2000 \
   --exchange-interval 500 --distance-output-interval 250 --report-interval 250 --traj-interval 500 \
   --production-phase-timers --platform-temp-directory ${TMPDIR:-/tmp} --us-allow-bad-windows"
# --us-allow-bad-windows (user-approved deviation from runbook, 2026-10-10): GA terminal CA-CA distance is rigid ~3.9 A,
# so W24 centres >= ~5.5 A fail the US start-quality gate (prep job 2898887). Applied uniformly to every leg;
# CSVs, model and slot table unchanged.
CMD_A="python -m gareus $F --run-mode cmd --windows-2d-csv scripts/cvaux_parity/W24.csv --aux-cv-model ${SCR}/M.json"
CMD_A_NOAUX="python -m gareus $F --run-mode cmd --windows-2d-csv scripts/cvaux_parity/W24_noaux.csv"

case "${MODE}" in
  prep)
    python -m gareus $F --platform CUDA --production-steps 500 --run-mode cmd \
       --windows-2d-csv scripts/cvaux_parity/W24_noaux.csv --out ${SCR}/RUN_P
    python scripts/cvaux_parity/make_model.py ${SCR}/RUN_P/01_solvated_start.pdb ${SCR}/M.json ;;
  cuda_mixed)  $CMD_A --platform CUDA --out ${SCR}/OUT_CUDA_MIXED
               scripts/cvaux_gpu_parity.sh ${SCR}/OUT_CUDA_MIXED > ${SCR}/parity_cuda_mixed.json ;;
  opencl)      $CMD_A --platform OpenCL --out ${SCR}/OUT_OPENCL
               scripts/cvaux_gpu_parity.sh ${SCR}/OUT_OPENCL > ${SCR}/parity_opencl.json ;;
  cuda_double) $CMD_A --platform CUDA --precision double --out ${SCR}/OUT_CUDA_DOUBLE
               scripts/cvaux_gpu_parity.sh ${SCR}/OUT_CUDA_DOUBLE > ${SCR}/parity_cuda_double.json ;;
  roundtrip)
    set +e
    GAREUS_TEST_FAIL_AT_PROD_STEP=3500 $CMD_A --platform CUDA --out ${SCR}/OUT_RT > >(tee ${SCR}/rt_leg1.out) 2> >(tee ${SCR}/rt_leg1.err >&2)
    rc=$?
    set -e
    sleep 2
    echo "[cvp] injected-failure leg exit code ${rc} (expected 1)"
    grep -q "GAREUS_TEST_FAIL_AT_PROD_STEP=3500: injected test failure" ${SCR}/rt_leg1.out ${SCR}/rt_leg1.err \
      || { echo "[cvp] ERROR: leg 1 did not die on the injected failure (rc=${rc}); not resuming"; exit 3; }
    env -u GAREUS_TEST_FAIL_AT_PROD_STEP $CMD_A --platform CUDA --out ${SCR}/OUT_RT --resume
    scripts/cvaux_gpu_parity.sh ${SCR}/OUT_RT > ${SCR}/parity_roundtrip.json ;;
  noaux)       $CMD_A_NOAUX --platform CUDA --out ${SCR}/OUT_BASE ;;
  *) echo "unknown mode ${MODE}"; exit 2 ;;
esac
echo "[cvp] ${MODE} done"

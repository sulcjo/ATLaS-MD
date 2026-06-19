#!/bin/bash
#SBATCH --job-name=aladip
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --output=aladip_%j.log
#SBATCH --error=aladip_%j.err
#
# Ace-Ala-Nme phi/psi validation (dense GaMD cumulant2).
# Submit FROM the repo root of the feat/ala-dipeptide-validation checkout:
#     sbatch ala_dipeptide.sbatch
# Requires: the modified GAREUS code (this branch). The run writes everything
# under RUNS/ala_dipeptide_validation/ (reference 100 ns, then validation
# 9-window hmr-gamd/REUS, 1,000,000 samples/window @ 0.1 ps, solute-only XTC),
# then runs the gamd_cumulant2 analysis + comparison.
set -Eeuo pipefail

# --- environment (matches RUN7.sh; adjust to your cluster) ---
source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc

# --- keep numpy/scipy BLAS from oversubscribing CPUs ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# --- CUDA kernel cache (avoids PTX recompile each restart) ---
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH="${HOME}/.cuda_kernel_cache"
mkdir -p "${CUDA_CACHE_PATH}"

# --- run from the submission directory (the repo root) ---
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "host=$(hostname) cwd=$(pwd) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import openmm; print('OpenMM', openmm.version.version, 'platforms', [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())])"

# --- run the full pipeline: fixture -> reference -> validation -> analysis -> compare ---
bash run_ala_dipeptide.sh

echo "FINISHED job ${SLURM_JOB_ID:-local}"

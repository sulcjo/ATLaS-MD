#!/bin/bash
#SBATCH --job-name=cvp_fastpath
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --output=/home/sulcjo/cvaux_parity_runs/logs/%x_%j.log
#SBATCH --error=/home/sulcjo/cvaux_parity_runs/logs/%x_%j.err
set -Eeuo pipefail
set --
CODE_DIR=/home/sulcjo/2026_peptide_sampler_cvaux_parity
cd "${CODE_DIR}"
source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH; export PYTHONPATH="${CODE_DIR}:${CODE_DIR}/tests"
export CUDA_CACHE_PATH="${TMPDIR:-/tmp}/cuda_kernel_cache_${SLURM_JOB_ID:-$$}"; mkdir -p "${CUDA_CACHE_PATH}"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
echo "[cvp] fastpath commit=$(cat DEPLOYED_COMMIT) host=$(hostname) job=${SLURM_JOB_ID}"
python scripts/cvaux_parity/fastpath_gpu.py /home/sulcjo/cvaux_parity_runs/fastpath_gpu.json

#!/bin/bash
#SBATCH --job-name=c8_integbench
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=1:00:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=/home/sulcjo/gareus/chignolin/c8_integbench_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/c8_integbench_%j.err
# 2026-09-22: is the 5x gap between Langevin (49.7 steps/s/rep) and chignolin_8 production
# (9.75) caused by the GaMD integrator's per-step host round trips? No MPS, same as production.
set -uo pipefail
ulimit -n 65536
source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc
export OPENMM_CPU_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH; export PYTHONPATH=/home/sulcjo/2026_peptide_sampler
export CUDA_CACHE_DISABLE=0 CUDA_CACHE_PATH="${TMPDIR:-/tmp}/cuda_kernel_cache_${SLURM_JOB_ID}"
mkdir -p "${CUDA_CACHE_PATH}"
T=/home/sulcjo/gareus/chignolin/c8_integ_bench.py
echo "host $(hostname) job ${SLURM_JOB_ID} | code $(cat /home/sulcjo/2026_peptide_sampler/DEPLOYED_COMMIT)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== single context per GPU (N=4): GPU-side cost without time-slicing"
for A in L L2 P B; do timeout 400 python "$T" "$A" 4 30 2>&1 | grep -E "RESULT|built|copied|Traceback|Error|Exception"; done
echo "=== 248 contexts, no MPS (production regime)"
for A in L L2 P B; do timeout 900 python "$T" "$A" 248 60 2>&1 | grep -E "RESULT|built|copied|Traceback|Error|Exception"; done
echo DONE

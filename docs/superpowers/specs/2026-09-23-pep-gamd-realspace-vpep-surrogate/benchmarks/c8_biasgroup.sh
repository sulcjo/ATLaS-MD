#!/bin/bash
#SBATCH --job-name=c8_biasgroup
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=1:15:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=/home/sulcjo/gareus/chignolin/c8_biasgroup_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/c8_biasgroup_%j.err
# 2026-09-22 lever 2 test. Previous job: is the 5x gap between Langevin (49.7 steps/s/rep) and chignolin_8 production
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
T=/home/sulcjo/gareus/chignolin/c8_integ_bench2.py
echo "host $(hostname) job ${SLURM_JOB_ID} | code $(cat /home/sulcjo/2026_peptide_sampler/DEPLOYED_COMMIT)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== lever 2: bias forces split (29+31, production) vs merged (29) -- real CV forces"
echo "--- 1 context/GPU (N=4), real Pep-GaMD integrator"
for B in split merged split merged; do BIAS=$B timeout 400 python "$T" P 4 30 2>&1 | grep -E "RESULT|Traceback|Error|Exception"; done
echo "--- 248 contexts, branch-free integrator (the real one segfaulted at 248 in job 2577764)"
for B in split merged; do BIAS=$B timeout 900 python "$T" B 248 60 2>&1 | grep -E "RESULT|built|Traceback|Error|Exception"; done
echo "--- 248 contexts, real integrator (retry; may segfault again)"
for B in split merged; do BIAS=$B timeout 900 python "$T" P 248 60 2>&1 | grep -E "RESULT|built|Traceback|Error|Exception"; echo "exit ${PIPESTATUS[0]}"; done
echo DONE

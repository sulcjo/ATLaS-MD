#!/bin/bash
#SBATCH --job-name=c8_mps59
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=1:30:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=/home/sulcjo/gareus/chignolin/c8_mps59_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/c8_mps59_%j.err
# chignolin_9 checklist T2 (2026-09-23): MPS throughput at 59 contexts/GPU with the REAL Pep-GaMD
# integrator in stage 5 (boost live, FSF printed), real CV forces (contact umbrella + residual CV2).
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
PIPE="${TMPDIR:-/tmp}/nvidia-mps-pipe_${SLURM_JOB_ID}"; MPSLOG="${TMPDIR:-/tmp}/nvidia-mps-log_${SLURM_JOB_ID}"
mps_start() { export CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG"; mkdir -p "$PIPE" "$MPSLOG"; nvidia-cuda-mps-control -d; sleep 2; }
mps_stop() { timeout 15 bash -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null; sleep 2; pkill -u $USER -f nvidia-cuda-mps 2>/dev/null; sleep 2; pkill -9 -u $USER -f nvidia-cuda-mps 2>/dev/null; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; rm -rf "$PIPE" "$MPSLOG"; }
trap 'mps_stop' EXIT
T=/home/sulcjo/gareus/chignolin/c8_integ_bench3.py
F="RESULT|built|Traceback|Error|Exception|could not be loaded"
echo "host $(hostname) job ${SLURM_JOB_ID} | nofile $(ulimit -Sn)"
echo "=== MPS on"
mps_start
echo "--- 236 contexts (59/GPU), PME stream disabled (the historical MPS setting)"
BIAS=split PME_DISABLE=true  timeout 900 python "$T" P5 236 60 2>&1 | grep -E "$F"
echo "--- 236 contexts (59/GPU), PME stream enabled (+8-11 % without MPS)"
BIAS=split PME_DISABLE=false timeout 900 python "$T" P5 236 60 2>&1 | grep -E "$F"
echo "--- 192 contexts (48/GPU)"
BIAS=split PME_DISABLE=false timeout 900 python "$T" P5 192 60 2>&1 | grep -E "$F"
echo "--- 64 contexts (16/GPU, chignolin_7 scale)"
BIAS=split PME_DISABLE=false timeout 600 python "$T" P5 64 60 2>&1 | grep -E "$F"
mps_stop
echo "=== MPS off, reference"
echo "--- 236 contexts, no MPS"
BIAS=split PME_DISABLE=false timeout 900 python "$T" P5 236 60 2>&1 | grep -E "$F"
echo DONE

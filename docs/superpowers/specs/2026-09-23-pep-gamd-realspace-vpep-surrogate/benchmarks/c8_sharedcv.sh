#!/bin/bash
#SBATCH --job-name=c8_sharedcv
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=1:15:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=/home/sulcjo/gareus/chignolin/c8_sharedcv_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/c8_sharedcv_%j.err
# chignolin_9 candidate T7 follow-up (2026-09-23): compute the 1,256-pair contact sum once per step.
# split  = production: CV1 contact umbrella (group 31) + residual-torsion-pc CV2 (group 29), each with
#          its own copy of the contact sum.
# shared = one CustomCVForce (group 29): the CV2 force's contact sub-CV also carries the CV1 umbrella.
# Real Pep-GaMD integrator in stage 5 (boost live), real CV forces from chignolin_8's frozen pair model.
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
T=/home/sulcjo/gareus/chignolin/c8_integ_bench4.py
F="RESULT|built|Traceback|Error|Exception|could not be loaded"
echo "host $(hostname) job ${SLURM_JOB_ID} | code $(cat /home/sulcjo/2026_peptide_sampler/DEPLOYED_COMMIT) | nofile $(ulimit -Sn)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo "=== equivalence: shared vs split bias energy/forces (mixed, then double)"
timeout 300 python "$T" CHECK 1 2>&1 | grep -E "$F"
CHECK_DOUBLE=1 timeout 300 python "$T" CHECK 1 2>&1 | grep -E "$F"
echo "=== no MPS, 1 context/GPU (N=4): per-context cost, PME stream enabled (no-MPS verdict)"
for B in split shared split shared; do BIAS=$B PME_DISABLE=false timeout 400 python "$T" P5 4 30 2>&1 | grep -E "$F"; done
echo "=== MPS on, 236 contexts (59/GPU, chignolin_9 layout), PME stream disabled (MPS verdict)"
mps_start
for B in split shared split shared; do BIAS=$B PME_DISABLE=true timeout 900 python "$T" P5 236 60 2>&1 | grep -E "$F"; done
mps_stop
echo DONE

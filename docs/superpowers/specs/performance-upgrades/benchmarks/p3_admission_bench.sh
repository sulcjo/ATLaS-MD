#!/bin/bash
#SBATCH --job-name=p3_admit
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=2:00:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --exclude=d094
#SBATCH --output=/home/sulcjo/gareus/chignolin/p3_admit_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/p3_admit_%j.err
# performance-upgrades P3 x P4: 236 resident contexts (59/GPU, chignolin_9 layout) under MPS,
# sweep active replicas per GPU (all/32/16/8) inside each process, one process per MPS
# active-thread percentage (inherit/50/25). Reference: shared layout, all admitted, MPS,
# PME stream disabled = 2,618-2,630 ns/day/node (job 2608721).
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
GPULOG=/home/sulcjo/gareus/chignolin/p3_admit_${SLURM_JOB_ID}_gpu.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 10 > "$GPULOG" 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null; mps_stop' EXIT
T=/home/sulcjo/gareus/chignolin/p3_admission_bench.py
F="RESULT|built|Traceback|Error|Exception|could not be loaded|AssertionError"
echo "host $(hostname) job ${SLURM_JOB_ID} | code $(cat /home/sulcjo/2026_peptide_sampler/DEPLOYED_COMMIT) | nofile $(ulimit -Sn)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
mps_start
for PCT in inherit 50 25; do
  echo "=== MPS on, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${PCT} (fresh process)"
  if [ "$PCT" = inherit ]; then
    ADMIT=all,32,16,8 REPS=2 PME_DISABLE=true timeout 1800 python "$T" 236 2>&1 | grep -E "$F"
  else
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$PCT ADMIT=all,16,8 REPS=2 PME_DISABLE=true timeout 1500 python "$T" 236 2>&1 | grep -E "$F"
  fi
done
mps_stop
echo DONE

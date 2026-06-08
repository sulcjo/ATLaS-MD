#!/bin/bash
#SBATCH --job-name=chigno6
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --ntasks-per-node=32
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:2
#SBATCH --output=chigno6_%j.log
#SBATCH --error=chigno6_%j.err

# --- environment ---
source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc

export PYTHONNOUSERSITE=1
unset PYTHONPATH

# --- CUDA kernel cache ---
# Avoids PTX recompilation on every restart; kernels are compiled once per
# topology/force combination and replayed from cache on subsequent runs.
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH="${HOME}/.cuda_kernel_cache"
mkdir -p "${CUDA_CACHE_PATH}"

# --- CUDA MPS (Multi-Process Service) ---
# Allows 8 replicas/GPU to share each GPU context concurrently rather than
# running sequentially, improving GPU utilization for small peptide replicas.
# Per-job dirs prevent conflicts when multiple SLURM jobs run on the same node.
# If GPUs are shared with other users or the cluster disables MPS, comment
# out the three lines below and remove --cuda-mps from the gareus invocation.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID:-$$}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${SLURM_JOB_ID:-$$}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d
echo "MPS started (pipe: ${CUDA_MPS_PIPE_DIRECTORY})"

# --- diagnostic ---
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

function cleanup {
    # Stop MPS before exit so the pipe dir can be cleaned up.
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    # Auto-resubmit: run continues with --resume until scancel stops the chain.
    sbatch --export=ALL,job_restarted=1 RUN6.sh
}
trap cleanup EXIT

cd /home/sulcjo/gareus

python -m gareus \
    --config chignolin_fulltreatment6.yaml \
    --platform CUDA \
    --device-index 0,1 \
    --precision mixed \
    --cuda-mps \
    --cuda-disable-pme-stream false \
    --platform-temp-directory "${TMPDIR:-/tmp}" \
    --cpu-budget 32 \
    --max-replicas 16 \
    --max-cpu-per-replica 2 \
    --out chignolin_2d_run6 \
    --resume

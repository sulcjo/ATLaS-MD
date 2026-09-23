#!/bin/bash
#SBATCH --job-name=gp_chignolin_9
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=64
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sulcjo/gareus/chignolin/genpept_chignolin_9_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/genpept_chignolin_9_%j.err

cd "/home/sulcjo/gareus/chignolin"

source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_PATH="${HOME}/.cuda_kernel_cache"
mkdir -p "${CUDA_CACHE_PATH}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# chignolin_9_genpept.yaml, NOT chignolin_9.yaml: the latter is the gareus production
# config (smoke/config/chignolin_9.yaml); aurum2's chignolin_7 generation reused one
# filename (chignolin.yaml) across both the genpept run and the later production
# config, which only works because the genpept output directory had already been
# written before the file was overwritten. Keeping the names distinct here avoids
# that trap entirely.
python /home/sulcjo/gareus/GENPEPT.py \
    --config "/home/sulcjo/gareus/chignolin/chignolin_9_genpept.yaml" \
    --platform CUDA \
    --min-jobs 4

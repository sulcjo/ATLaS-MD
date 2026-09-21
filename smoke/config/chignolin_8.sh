#!/bin/bash
#SBATCH --job-name=chignolin_8
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=4:00:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:4
#SBATCH --output=/home/sulcjo/gareus/chignolin/chignolin_8_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/chignolin_8_%j.err
#SBATCH --exclusive
set -Eeuo pipefail

# chignolin_8: lambda-ladder adaptive production with AUTOMATIC CV2 SELECTION
# (cvs.cv2: auto in chignolin_8.yaml). Same chain mechanics as chignolin_7.sh.
# Epoch 0 IS the unbiased swarm: it grafts members from the GENPEPT library, pools
# the frozen Pep-GaMD envelope, fits residual torsion components orthogonal to the
# contact anchor, freezes the selected one as swarm/analysis/cv_pair_model.json
# (+ cv_candidate_set.json, cv_feature_schema.json) and writes a CV1 x CV2 x lambda
# ladder plus ladder_run_args.yaml carrying cv2: residual-torsion-pc and the three
# artifact paths. Epochs 1..N sample that ladder; the pair is never redefined
# (tica_switch_cv2 is forced off, a resume with a different model refuses).
# Selection verdict lands in swarm/analysis/cv_selection_report.json (status,
# coupling fraction, CV1-width shrink, per-rung CV2 ESS) -- read it before the
# pool is spent. Differences from chignolin_7: gamd recon at ns scale with stage 2
# on (see the yaml), cv2 auto, cv_selection block, swarm_n_windows_cv2 4.
# Seed library chignolin_genpept_rep5_t7 was generated with the 'chignolin'
# GENPEPT preset (fold-aware banks); the pair model records that, so this run is
# NOT quotable as ab initio (user ruling 2026-09-21).
# Code: ~/2026_peptide_sampler (rsync deploy of main; commit in DEPLOYED_COMMIT).
# --mem 192G (was 96G): max_replicas 256 = 4x chignolin_7's 64 contexts in one process;
# both partitions have >= 384 GB nodes and the job is --exclusive anyway.
PEPTIDE="chignolin_8"
CODE_DIR="/home/sulcjo/2026_peptide_sampler"
RUN_DIR="/home/sulcjo/gareus/chignolin"
CONFIG="${RUN_DIR}/chignolin_8.yaml"
OUT_DIR="${RUN_DIR}/chignolin_8"
POOL_JSON="${OUT_DIR}/adaptive_production/adaptive_runtime_pool.json"
DRIVER_SUMMARY_JSON="${OUT_DIR}/adaptive_production/adaptive_production_driver_summary.json"
RUN_SCRIPT="${RUN_DIR}/chignolin_8.sh"
# Safety net for unattended running: the chain only resubmits on a walltime stop, but
# nothing otherwise bounds it. 6000 ns at ~2500 ns/day is ~58 h = ~15 four-hour jobs;
# 30 is generous headroom and still stops a runaway.
MAX_RESUBMITS=50
# Unattended chains die silently: every cleanup() exit path shares one `return`, so a
# 03:00 stop looks identical whether the pool was spent or the cap tripped. Stamp the
# reason into a marker file so a morning `cat` answers it without log archaeology.
CHAIN_STATUS="${RUN_DIR}/chignolin_8_CHAIN_STATUS.txt"
function chain_marker {
    printf '%s | job %s | %s\n' "$(date -Is)" "${SLURM_JOB_ID:-none}" "$1" >> "${CHAIN_STATUS}"
}
cd "${RUN_DIR}"

STOP_REQUESTED=0
GAREUS_PID=""
MPS_STARTED=0

function pool_is_spent {
    [[ ! -f "${POOL_JSON}" ]] && return 1
    python3 - "${POOL_JSON}" <<'PYPOOL'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
total = float(d.get("total_ns", 0))
used  = float(d.get("used_ns",  0))
sys.exit(0 if total > 0 and used >= total * 0.99 else 1)
PYPOOL
}

function driver_is_complete {
    [[ ! -f "${DRIVER_SUMMARY_JSON}" ]] && return 1
    python3 - "${DRIVER_SUMMARY_JSON}" <<'PYDRIVER'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
sys.exit(0 if d.get("status") == "completed" else 1)
PYDRIVER
}

function has_any_progress {
    [[ -f "${OUT_DIR}/01_solvated_start.pdb" ]] && return 0
    [[ -f "${OUT_DIR}/00_built_peptide.pdb" ]] && return 0
    return 1
}

function request_graceful_stop {
    STOP_REQUESTED=1
    echo "[gareus] Stop signal for ${PEPTIDE}; forwarding SIGTERM so GAREUS writes a checkpoint."
    if [[ -n "${GAREUS_PID:-}" ]] && kill -0 "${GAREUS_PID}" 2>/dev/null; then
        kill -TERM "${GAREUS_PID}" 2>/dev/null || true
    fi
}

function cleanup {
    local exit_code=$?
    if [[ "${MPS_STARTED:-0}" == "1" ]]; then
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    fi
    rm -rf "${CUDA_MPS_PIPE_DIRECTORY:-}" "${CUDA_MPS_LOG_DIRECTORY:-}" "${CUDA_CACHE_PATH:-}" 2>/dev/null || true
    if pool_is_spent; then
        echo "[gareus] MD pool spent for ${PEPTIDE}; chain ends."
        chain_marker "DONE: md pool spent"
        return
    fi
    if driver_is_complete; then
        echo "[gareus] Adaptive-production driver reports status=completed for ${PEPTIDE}; chain ends."
        chain_marker "DONE: driver status=completed"
        return
    fi
    if [[ "${exit_code}" -ne 0 && "${STOP_REQUESTED:-0}" != "1" ]]; then
        echo "[gareus] ${PEPTIDE} exited with code ${exit_code} unexpectedly; no resubmission."
        chain_marker "FAILED: exit code ${exit_code}, chain stopped -- inspect logs"
        return
    fi
    local n="${resubmit_count:-0}"
    if (( n >= MAX_RESUBMITS )); then
        echo "[gareus] resubmit cap ${MAX_RESUBMITS} reached; chain ends. Pool not spent -- inspect before continuing."
        chain_marker "STOPPED: resubmit cap ${MAX_RESUBMITS} reached, pool NOT spent"
        return
    fi
    echo "[gareus] Resubmitting ${PEPTIDE} with --resume (count $((n+1)) of ${MAX_RESUBMITS})."
    if ! sbatch --export=ALL,job_restarted=1,resubmit_count=$((n+1)) "${RUN_SCRIPT}"; then
        echo "[gareus] ERROR: sbatch resubmission failed for ${PEPTIDE}."
        chain_marker "FAILED: sbatch resubmission rejected"
        exit 92
    fi
}

trap request_graceful_stop TERM USR1
trap cleanup EXIT

# --- preflight: this run designs its own state space in epoch 0, so there are no
#     swarm artefacts to check for. What it cannot do without is the GENPEPT seed
#     library epoch 0 grafts its members from, and the config itself.
GENPEPT_DIR="$(python3 - "${CONFIG}" <<'PYSEED'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
print((d.get("starting_structures") or {}).get("seed_conformers_dir", ""))
PYSEED
)"
[[ -n "${GENPEPT_DIR}" ]] || { echo "[gareus] ERROR: starting_structures.seed_conformers_dir is not set in ${CONFIG}."; exit 90; }
[[ -s "${GENPEPT_DIR}/final_survivor_seeds.csv" ]] || {
    echo "[gareus] ERROR: no GENPEPT library at ${GENPEPT_DIR} (need final_survivor_seeds.csv)."
    echo "[gareus]        Epoch 0 grafts every swarm member from it; it cannot start without one."
    exit 90
}
echo "[gareus] preflight OK: GENPEPT library ${GENPEPT_DIR}"
# If a previous job already got epoch 0 through its gate, say so -- the run will
# skip straight to epoch 1 rather than re-running any of it.
if [[ -f "${OUT_DIR}/swarm/analysis/epoch0_complete.json" ]]; then
    echo "[gareus] epoch 0 already complete; resuming at the ladder it designed."
    if [[ -f "${OUT_DIR}/swarm/analysis/cv_selection_report.json" ]]; then
        python3 - "${OUT_DIR}/swarm/analysis/cv_selection_report.json" <<'PYCV'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[gareus] auto CV2: status={d.get('status')} preset={d.get('genpept_preset')} "
      f"component={d.get('selected_component_index')} layout={(d.get('layout') or {}).get('kind')} "
      f"coupling_fraction={(d.get('certificate') or {}).get('coupling_fraction_of_k1')}")
PYCV
    fi
elif [[ -d "${OUT_DIR}/swarm" ]]; then
    echo "[gareus] epoch 0 in progress; finished members will be skipped."
fi

# --- environment ---
source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc

# Each OpenMM Context builds a CPU thread pool sized to the core count (192 here),
# so a 112-window ladder wanted ~43,000 threads and aborted at context ~72 with
# 'terminate called without an active exception' -- GPU memory was only 9 of 46 GB.
# PME runs on the GPU (UseCpuPme unset -> OpenMM default false, DisablePmeStream
# true), so that pool is idle infrastructure. Measured on this node: capping it
# takes a context from 194 threads to 3, GPU memory and RSS unchanged.
export OPENMM_CPU_THREADS=8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="${CODE_DIR}"
export PYTHONFAULTHANDLER=1
# Python block-buffers stdout when it is a file, so ~8 KB of progress sat
# unwritten and a SIGTERM at walltime discarded it -- the log looked dead
# while the run was fine. Unbuffered costs nothing here and makes the log
# usable for watching an unattended chain.
export PYTHONUNBUFFERED=1

export CUDA_CACHE_DISABLE=0
# JIT kernel cache on node-local scratch, NOT NFS $HOME (flock deadlock on first context).
export CUDA_CACHE_PATH="${TMPDIR:-/tmp}/cuda_kernel_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${CUDA_CACHE_PATH}"
# MPS: many replica contexts share 4 GPUs; --cuda-mps below assumes a real daemon.
export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/nvidia-mps-pipe_${SLURM_JOB_ID:-$$}"
export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/nvidia-mps-log_${SLURM_JOB_ID:-$$}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d
MPS_STARTED=1

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
echo "[gareus] code commit: $(cat ${CODE_DIR}/DEPLOYED_COMMIT)  host: $(hostname)  job: ${SLURM_JOB_ID}"

# --- resume: filesystem-based ---
if has_any_progress; then
    RESUME_FLAG="--resume"
    echo "[gareus] Existing output at ${OUT_DIR}; resuming ${PEPTIDE}."
elif [[ "${job_restarted:-0}" == "1" ]]; then
    echo "[gareus] ERROR: job_restarted=1 but no output at ${OUT_DIR}; refusing fresh restart."
    exit 90
else
    RESUME_FLAG=""
    echo "[gareus] No prior output; starting ${PEPTIDE} from scratch."
fi

python -m gareus \
    --config "${CONFIG}" \
    --out "${OUT_DIR}" \
    --platform CUDA \
    --device-index 0,1,2,3 \
    --precision mixed \
    --cuda-disable-pme-stream true \
    --cuda-mps \
    --platform-temp-directory "${TMPDIR:-/tmp}" \
    --tui-mode dashboard \
    --progress-mode both \
    --us-pull-workers 28 \
    --us-start-primary-bad-bias-kcal 15.0 \
    ${RESUME_FLAG} &
GAREUS_PID=$!
set +e
wait "${GAREUS_PID}"
GAREUS_EXIT=$?
set -e
exit "${GAREUS_EXIT}"

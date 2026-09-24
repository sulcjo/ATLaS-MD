#!/bin/bash
#SBATCH --job-name=chignolin_9
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --ntasks-per-node=192
#SBATCH --time=4:00:00
#SBATCH --partition=d192_384_gpu,d192_768_gpu
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:4
#SBATCH --output=/home/sulcjo/gareus/chignolin/chignolin_9_%j.log
#SBATCH --error=/home/sulcjo/gareus/chignolin/chignolin_9_%j.err
#SBATCH --exclusive
set -Eeuo pipefail

# chignolin_9: 236-state lambda ladder (59 CV centres x 4 rungs) run under CUDA MPS,
# 59 contexts/GPU. Chain mechanics identical to chignolin_8.sh; the MPS launcher
# section below is new -- see docs/atlas-md/developer/gpu-throughput-benchmark-todo.md
# ("DECIDED 2026-09-22: chignolin_9 runs 236 states under MPS (59 contexts/GPU)").
# Decision basis: MPS refuses >~60 client contexts/GPU on these L40S (c8_ctxtest3
# job 2563506), so 248 without MPS was the only way to keep chignolin_8's full state
# count; chignolin_9 instead narrows the state count to fit under MPS and reclaims
# the MPS throughput (measured T2, job 2580889: 2,307 ns/day/node @ 59/GPU real
# integrator stage 5 = 2.75x the 840 ns/day/node no-MPS reference; node aggregate is
# CPU co-limited at 59/GPU, not the 3,154 ns/day @16/GPU benchmark ceiling).
# STILL OPEN when this script was written (do not launch before these land):
#   - state-space item: regenerate the CV2 layout to exactly 59 centres (swarm
#     analyze / ladder design); chignolin_9.yaml does not exist yet.
#   - pull-workers-under-MPS is UNVERIFIED: T2's 2,307 ns/day/node measured
#     production-only contexts under MPS; it never combined the up-to-28
#     --us-pull-workers pull contexts (opened by this same process, round-robined
#     over --us-pull-device-index 0,1,2,3, ~7/GPU) running concurrently with the 59
#     production contexts already open on a GPU. If a future run shows pull-phase
#     contexts are not fully closed before production opens its 59/GPU, either drop
#     --us-pull-workers per-GPU below (~60 - 59) or start MPS only after the pull
#     phase completes (see cleanup ordering below) rather than at job start.
# Code: ~/2026_peptide_sampler (rsync deploy of main; commit in DEPLOYED_COMMIT).
PEPTIDE="chignolin_9"
CODE_DIR="/home/sulcjo/2026_peptide_sampler"
RUN_DIR="/home/sulcjo/gareus/chignolin"
CONFIG="${RUN_DIR}/chignolin_9.yaml"
OUT_DIR="${RUN_DIR}/chignolin_9"
POOL_JSON="${OUT_DIR}/adaptive_production/adaptive_runtime_pool.json"
DRIVER_SUMMARY_JSON="${OUT_DIR}/adaptive_production/adaptive_production_driver_summary.json"
RUN_SCRIPT="${RUN_DIR}/chignolin_9.sh"
# Safety net for unattended running: the chain only resubmits on a walltime stop, but
# nothing otherwise bounds it.
MAX_RESUBMITS=50
CHAIN_STATUS="${RUN_DIR}/chignolin_9_CHAIN_STATUS.txt"
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
    echo "[ATLaS-MD] Stop signal for ${PEPTIDE}; forwarding SIGTERM so ATLaS-MD writes a checkpoint."
    if [[ -n "${GAREUS_PID:-}" ]] && kill -0 "${GAREUS_PID}" 2>/dev/null; then
        kill -TERM "${GAREUS_PID}" 2>/dev/null || true
    fi
}

function cleanup {
    local exit_code=$?
    if [[ "${MPS_STARTED:-0}" == "1" ]]; then
        timeout 15 bash -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null || true
        sleep 2
        pkill -u "$USER" -f nvidia-cuda-mps 2>/dev/null || true
        sleep 2
        pkill -9 -u "$USER" -f nvidia-cuda-mps 2>/dev/null || true
    fi
    rm -rf "${CUDA_MPS_PIPE_DIRECTORY:-}" "${CUDA_MPS_LOG_DIRECTORY:-}" "${CUDA_CACHE_PATH:-}" 2>/dev/null || true
    if pool_is_spent; then
        echo "[ATLaS-MD] MD pool spent for ${PEPTIDE}; chain ends."
        chain_marker "DONE: md pool spent"
        return
    fi
    if driver_is_complete; then
        echo "[ATLaS-MD] Adaptive-production driver reports status=completed for ${PEPTIDE}; chain ends."
        chain_marker "DONE: driver status=completed"
        return
    fi
    if [[ "${exit_code}" -ne 0 && "${STOP_REQUESTED:-0}" != "1" ]]; then
        echo "[ATLaS-MD] ${PEPTIDE} exited with code ${exit_code} unexpectedly; no resubmission."
        chain_marker "FAILED: exit code ${exit_code}, chain stopped -- inspect logs"
        return
    fi
    local n="${resubmit_count:-0}"
    if (( n >= MAX_RESUBMITS )); then
        echo "[ATLaS-MD] resubmit cap ${MAX_RESUBMITS} reached; chain ends. Pool not spent -- inspect before continuing."
        chain_marker "STOPPED: resubmit cap ${MAX_RESUBMITS} reached, pool NOT spent"
        return
    fi
    echo "[ATLaS-MD] Resubmitting ${PEPTIDE} with --resume (count $((n+1)) of ${MAX_RESUBMITS})."
    if ! sbatch --export=ALL,job_restarted=1,resubmit_count=$((n+1)) "${RUN_SCRIPT}"; then
        echo "[ATLaS-MD] ERROR: sbatch resubmission failed for ${PEPTIDE}."
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
[[ -n "${GENPEPT_DIR}" ]] || { echo "[ATLaS-MD] ERROR: starting_structures.seed_conformers_dir is not set in ${CONFIG}."; exit 90; }
[[ -s "${GENPEPT_DIR}/final_survivor_seeds.csv" ]] || {
    echo "[ATLaS-MD] ERROR: no GENPEPT library at ${GENPEPT_DIR} (need final_survivor_seeds.csv)."
    echo "[ATLaS-MD]        Epoch 0 grafts every swarm member from it; it cannot start without one."
    exit 90
}
echo "[ATLaS-MD] preflight OK: GENPEPT library ${GENPEPT_DIR}"
if [[ -f "${OUT_DIR}/swarm/analysis/epoch0_complete.json" ]]; then
    echo "[ATLaS-MD] epoch 0 already complete; resuming at the ladder it designed."
elif [[ -d "${OUT_DIR}/swarm" ]]; then
    echo "[ATLaS-MD] epoch 0 in progress; finished members will be skipped."
fi

# --- environment ---
source /uochb/soft/a/spack/20221129-git/share/spack/setup-env.sh
source /home/sulcjo/miniforge/current/bin/activate
eval "$(mamba shell hook --shell bash)"
mamba activate /home/sulcjo/conda-envs/calc

export OPENMM_CPU_THREADS=8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="${CODE_DIR}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

export CUDA_CACHE_DISABLE=0
# JIT kernel cache on node-local scratch, NOT NFS $HOME (flock deadlock on first context).
export CUDA_CACHE_PATH="${TMPDIR:-/tmp}/cuda_kernel_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${CUDA_CACHE_PATH}"
export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/nvidia-mps-pipe_${SLURM_JOB_ID:-$$}"
export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/nvidia-mps-log_${SLURM_JOB_ID:-$$}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
# Same descriptor-exhaustion fix as chignolin_8 (SLURM propagates a 1024 login soft
# limit; the CUDA driver only raises it to 4096, and MPS clients cost ~18 fds each).
# Raise it before the MPS daemon (which inherits it) and before python.
ulimit -n 65536
echo "[ATLaS-MD] nofile soft/hard: $(ulimit -Sn)/$(ulimit -Hn)"

# MPS ON for this campaign (2026-09-22 decision): 236 states = 59 contexts/GPU, under
# the ~60/GPU MPS client ceiling measured on chignolin_8 (c8_ctxtest3 job 2563506).
echo "[ATLaS-MD] starting CUDA MPS control daemon"
nvidia-cuda-mps-control -d
sleep 2
MPS_STARTED=1

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
echo "[ATLaS-MD] code commit: $(cat ${CODE_DIR}/DEPLOYED_COMMIT)  host: $(hostname)  job: ${SLURM_JOB_ID}"

# --- resume: filesystem-based ---
if has_any_progress; then
    RESUME_FLAG="--resume"
    echo "[ATLaS-MD] Existing output at ${OUT_DIR}; resuming ${PEPTIDE}."
elif [[ "${job_restarted:-0}" == "1" ]]; then
    echo "[ATLaS-MD] ERROR: job_restarted=1 but no output at ${OUT_DIR}; refusing fresh restart."
    exit 90
else
    RESUME_FLAG=""
    echo "[ATLaS-MD] No prior output; starting ${PEPTIDE} from scratch."
fi

# --cuda-disable-pme-stream true: measured under MPS at 236 contexts (59/GPU), real
# integrator stage 5 -- disabled 2,307 ns/day/node vs enabled 2,220 (c8_mps59, job
# 2580889); reproduced at 2,290 with it disabled in job 2608721. Opposite of the no-MPS
# verdict (enabled +8-11 %).
# --cuda-use-blocking-sync: chignolin_8's A/B (true, job 2567463) was measured
# without MPS, where the driver, not MPS, schedules the 236 client threads across
# 192 cores; unverified whether it still wins under MPS. Left true (safe default).
python -m gareus \
    --config "${CONFIG}" \
    --out "${OUT_DIR}" \
    --platform CUDA \
    --device-index 0,1,2,3 \
    --precision mixed \
    --cuda-mps \
    --cuda-disable-pme-stream true \
    --cuda-use-blocking-sync true \
    --cuda-deterministic-forces false \
    --platform-temp-directory "${TMPDIR:-/tmp}" \
    --tui-mode dashboard \
    --progress-mode both \
    --us-pull-workers 28 \
    --us-pull-device-index 0,1,2,3 \
    --us-start-primary-bad-bias-kcal 15.0 \
    ${RESUME_FLAG} &
GAREUS_PID=$!
set +e
# A trapped signal makes bash return from `wait` immediately while python is still
# alive; the script would then exit, SLURM would tear the job down, and the
# graceful-shutdown checkpoint would never be written (job 2563608, 2026-09-22,
# chignolin_8). Re-wait until the PID is really gone.
wait "${GAREUS_PID}"
GAREUS_EXIT=$?
while kill -0 "${GAREUS_PID}" 2>/dev/null; do
    wait "${GAREUS_PID}"
    GAREUS_EXIT=$?
done
set -e
exit "${GAREUS_EXIT}"

#!/usr/bin/env bash
# Print parity_report JSON for one or more finished aux run dirs. Read-only; launches no jobs.
# Usage: scripts/cvaux_gpu_parity.sh RUN_DIR [RUN_DIR ...]   (run from a CODE_DIR checkout)
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $0 RUN_DIR [RUN_DIR ...]" >&2; exit 2; }
for d in "$@"; do [ -d "$d" ] || { echo "not a directory: $d" >&2; exit 2; }; done
cd "$(dirname "$0")/.."
exec python -m gareus.auxiliary_cv.parity_report "$@"

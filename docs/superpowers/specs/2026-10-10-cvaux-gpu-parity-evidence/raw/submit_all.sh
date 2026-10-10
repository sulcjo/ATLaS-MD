#!/bin/bash
# Submit the whole CVaux GPU parity set: prep, then 5 runs afterok:prep.
set -euo pipefail
SCR=/home/sulcjo/cvaux_parity_runs
J=/home/sulcjo/2026_peptide_sampler_cvaux_parity/scripts/cvaux_parity/parity_job.sh
mkdir -p ${SCR}/logs
cd ${SCR}
P=$(sbatch --parsable --job-name=cvp_prep -o logs/%x_%j.log -e logs/%x_%j.err $J prep)
echo "prep ${P}"
for m in cuda_mixed opencl cuda_double roundtrip noaux; do
  id=$(sbatch --parsable --dependency=afterok:${P} --job-name=cvp_${m} -o logs/%x_%j.log -e logs/%x_%j.err $J ${m})
  echo "${m} ${id}"
done

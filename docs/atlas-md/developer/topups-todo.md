# TODO — adaptive top-ups follow-ups (opened 2026-09-24)

Design: `docs/superpowers/specs/2026-09-24-effective-topups-design.md`. Top-ups are off by default and off in
chignolin_9 (`ap_topups: false`).

## Synthetic validation (done, 2026-09-25)

Spec section 6 (Task 10, controller rulings 22-31). Ran a 20-seed A/B study over four analytic 2D
landscapes (a 112-state, 28-centre x 4-rung ladder) driving the real allocator, union-MBAR diagnostics,
throughput model and layout graphs. Result: **no landscape has a budget where the design's heterogeneous
deficit regime exists** — at the rows floor the split-halves test needs (median >= 40 decorrelated rows per
state), every uniform-baseline (top-ups-off) state is already at or under the sigma target, so spec 6.2's
heterogeneous-vs-uniform criterion is untestable in this harness. Reported for information only, at the rows
floor: top-ups lower the worst per-state sigma on two of four landscapes (gated-barrier -8.4%, 20/20 seeds,
p=9.5e-7; slow-cv2-double-branch -1.3%, 19/20) but improve PMF RMSE on none (gated-barrier 7% worse, 7/20
seeds). Missing-bridge routing (an edge spanning a genuine gap is routed as structural; the same coordinate
pair in an intact layout is not) passes 20/20. The patch τ penalty (kept deliberately — it is a real cost of
a lockstep patch with fewer exchange partners than the full state set) measured g(patch)/g(all-state) =
1.19-1.26. **Honest summary: top-ups are not shown to help in this harness.** Full numbers and regenerate
commands: `docs/superpowers/specs/2026-09-24-effective-topups/synth_study/README.md`. chignolin_10 (below)
remains the real test.

## T1 — chignolin_10: real-MD test of the new top-ups (not started)

chignolin_10 proceeds as the real-MD test regardless (user decision 2026-09-24): the synthetic harness found
no targetable heterogeneous-deficit regime on any of its four landscapes, so it neither supports nor blocks
running chignolin_10 — it is simply uninformative here, not a green light or a red flag. Compare against a
uniform-extension control at equal
wall-hours: same layout, lambda ladder and budget; same contexts/GPU in both arms (or report the targeting
and throughput effects separately); paired seeds for the shared baseline and distinct RNG streams after the
branch; replicates per arm (min per-state ESS is a noisy order statistic). Success: lower max per-state
free-energy uncertainty and block-bootstrap PMF uncertainty, no worse wall-clock per ns, no increase in dead
exchange pairs.

## T2 — throughput curve T(n) on the production node (not started)

The allocator's cost model uses a provisional two-point table from chignolin_8 (16/GPU = 3,154, 59/GPU =
~2,300 ns/day/node). Measure 4, 8, 16, 24, 32, 48, 59 contexts/GPU with the production integrator under MPS
and replace `ap_topup_throughput_table`.

## T3 — real-file seeding spike (not started)

Load chignolin_9 `epoch_001/baseline` binary checkpoints into top-up contexts for the same windows, re-apply
window/lambda globals, and check the seeding assertion (seeded frame's reduced potential == last logged u_k).

## T4 — sequential-panel benchmark (not started)

The small board's largest efficiency lever: run the campaign as sequential ~16 ctx/GPU panels instead of one
236-context run (~0.73x wall-time for the same per-state ns if throughput holds). Cost: no exchange across
panels. Measure realised per-state autocorrelation in panel vs full-graph mode before deciding.

## T5 — re-measure the 0.15/0.25 rung-overlap thresholds on the pairwise scale (not started)

Campaign-end rung overlap (the quality gate's `add_rung` and `analyze_gareus_mbar.py`'s ladder health check)
switched from a full-union MBAR overlap matrix to a pairwise statistic (commit `f7e007b`). The existing
`min_rung_overlap` (0.15) and `target_rung_overlap` (0.25) thresholds were calibrated on the old full-union
scale — real data shows the two scales disagree by roughly 2.7x on a large ladder (chignolin_7, 64 states:
median 0.089 full-union vs 0.258 pairwise on the same 48 adjacent-rung edges). Re-measure both thresholds
directly on the pairwise statistic before trusting a rung-health verdict on a new large ladder campaign.

## T6 — union-diagnostics memory before chignolin_10 (not started)

The per-epoch union-MBAR solve top-ups add (`union_diagnostics_from_npz`, called up to twice per
top-ups-on phase) measured 102.7 s / 14.6 GB peak RSS at 1,000,000 rows and 26.4 s / 4.06 GB at 250,000 rows,
at 236 states. This runs on the driver/analysis node, not a GPU, and an out-of-memory kill during the solve
cannot be caught. Before enabling `--ap-topups` on chignolin_10 (or any campaign of comparable size), either
subsample the union input before the solve or confirm the launch node has enough RAM for the campaign's full
row count.

## T7 — compress and prune output files before chignolin_10 (implemented; roll-out pending)

Spec: `docs/superpowers/specs/2026-09-28-output-retention-design.md` (28 Sep: 688 GB, 92 % checkpoint generations, of which only the newest per phase is ever read).

Requested 2026-09-27. chignolin_9's run directory is 334 GB (local copy of 27 Sep, `final/baseline` at 28 %;
aurum2 reports 188 GB for the same 282 GB of files because its filesystem compresses). Where it goes:

| Output | Size | Note |
| --- | --- | --- |
| Checkpoint generations (`*/checkpoints/generations/*/replica_*.chk`) | 328 GB | every generation kept: 30-67 per phase x 236 replicas x 6.9 MB (~1.6 GB/generation); epoch_001/baseline alone 108 GB |
| PDB files | 15 GB | seed banks, `final_pdbs`, `filtered_seed_bank` (~700-1,100 PDBs per directory) |
| `progress.jsonl` | 11 GB | one growing JSON line per progress/distances event, uncompressed |
| Trajectories (`.xtc`), samples/exchanges (Parquet) | 4 GB | already compressed formats |

To do before launching chignolin_10:

1. **Checkpoint retention (biggest win).** Keep only the newest N generations per phase (e.g. 2-3, enough for
   a corrupt-latest fallback) and delete older ones once a newer generation is fully written and verified.
   Check resume, `load_production_checkpoint`, top-up seeding (`final_window_states/`) and the Pep-GaMD
   integrator-globals restore do not read old generations first.
2. **`progress.jsonl`.** Rotate per job or per phase and gzip closed files, or stop writing a line for every
   `distances` event (it duplicates the Parquet samples). Check readers (dashboard, `gareus_monitor.py`,
   resume ETA) before changing the format.
3. **PDB sets.** Write seed banks and `final_pdbs` as one compressed archive or a multi-model/`.pdb.gz` file;
   check `GENPEPT.py`, seeding and analysis readers accept it.
4. **Checkpoints themselves.** Measure whether gzip/zstd on `.chk` is worth the resume latency (binary
   OpenMM state, likely compresses poorly; measure before deciding).

Measure the saving on a copy of chignolin_9 before changing defaults; none of this may change sampled data.

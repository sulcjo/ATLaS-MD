# TODO — adaptive top-ups follow-ups (opened 2026-09-24)

Design: `docs/superpowers/specs/2026-09-24-effective-topups-design.md`. Top-ups are off by default and off in
chignolin_9 (`ap_topups: false`); the new top-ups are validated synthetically first (spec section 6). The
items below are the real-world steps deliberately left out of that spec.

## T1 — chignolin_10: real-MD test of the new top-ups (not started)

Run only after the synthetic validation passes. Compare against a uniform-extension control at equal
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

# Swarm Stage Hand-off: S1 → S2/S3/S4

Unbiased swarm stage produces frozen lambda-ladder, seed bank, and shared GaMD envelope for production. Assume `cwd = <repo>`, r7 GENPEPT seed library deployed.

## Round 0: run members

```bash
gareus --config examples/chignolin_swarm_stage.yaml \
  --seed-conformers-dir <r7-dir> --out <run>/ --swarm-stage run
```
Prints `n_cells, R, n_members, budget_ns`. Output: `<run>/swarm/round_000/{plan.csv, member_XXXX/*}`.

## Analyze: pool, design ladder, gate

```bash
gareus --config examples/chignolin_swarm_stage.yaml \
  --seed-conformers-dir <r7-dir> --out <run>/ --swarm-stage analyze
```
On `swarm_report.json["gate"]["fail"]`, extend with `--swarm-replicates-per-cell $((R + extra)) --swarm-round 0` and re-analyze.

Output: `<run>/swarm/analysis/{envelope_discard.json, shared_gamd_setup/shared_gamd_setup_globals.json, ladder_design.json, windows_lambda_ladder.csv, seed_bank/final_survivor_seeds.csv, swarm_gate.json, swarm_report.json, ladder_run_args.yaml}`.

## Compare with S3 pilot: freeze envelope

```bash
gareus --config examples/chignolin_swarm_stage.yaml \
  --seed-conformers-dir <r7-dir> --out <run>/ --swarm-stage compare \
  --swarm-pilot-globals <pilot-shared-gamd-setup-globals.json>
```
Must have `pilot_comparison.json["freeze_allowed"] == true`. Fail → extend swarm, never hand-edit envelope.

## Production: consume ladder + seeds

Use: `--windows-2d-csv <run>/swarm/analysis/windows_lambda_ladder.csv --seed-conformers-dir <run>/swarm/analysis/seed_bank --shared-gamd-setup-dir <run>/swarm/analysis/shared_gamd_setup` + chignolin_7 pull settings (`us_pull_steps_per_window 150000 us_pull_timestep_fs 3.0 us_pull_k 300 us_pull_ramp_stages 10`). Or merge `ladder_run_args.yaml` into campaign YAML and use `--config` twice.

**Every window start grafts a seed and runs pull** (cli.py:471–478); missing `--seed-conformers-dir` fails all contact-window gates (S3 pilot, 2026-09-07). Budget: set `gamd_cmd_steps: 500000` (2 ns at 4 fs ≈ 7 min on L40S); this is a safety check, envelope is from `--shared-gamd-setup-dir`. **Check `ladder_design.json["fsf_floor_per_rung"]`**: warn if top rung floor < 0.5 (Pep-GaMD FSF unclamped; 100 ps cMD at k0 = 1 crashes).

## S5 re-seeding (round ≥ 1)

```bash
gareus --config examples/chignolin_swarm_stage.yaml \
  --seed-conformers-dir <r7-dir> --out <run>/ --swarm-stage run \
  --swarm-round 1 --swarm-seed-source production-frames \
  --swarm-production-seed-csv <frames.csv>
```
Envelope/ladder frozen; only adds seeds and diagnostics.

## Spec replacements

- `§3.5 M` (rung spacing) ← `ladder_design.json`; `§3.5 spacing` ← same
- `§12.1–3` (R, discard, tracking) ← `plan.csv`, `envelope_discard.json`, `ladder_design.json`

## Open items (not fixed here)

1. Graft summary `CV_after` prints ~10 Å for contacts CV (distance fallback); reporting defect only.
2. `finalize_run_manifest` not called on swarm dispatch exception path (cli.py).
3. `compare_envelopes` divides by pilot σ/k0 without zero-check (pilot may lack channels).
4. FSF clamping at zero with linear boost extension is method change; not adopted here (user decision).
5. Round ≥ 1 requires `--seed-conformers-dir` despite reading production frames.
6. Driver loads seed library twice on fresh build.

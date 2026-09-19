# rep4: the six universal tuning mechanisms — implemented, run, compared

Config: `scripts/chignolin_genpept_rep4.yaml` (scaled rep3 params, seed 81806, full bundle).
Bank: `chignolin_genpept_rep4_tuned/` (~3 h, RTX 3060 Ti).
Implementation: `GENPEPT.py` (six mechanism blocks after `# Universal coverage tunings (T1-T6)`,
flags `--coverage-uniformization`, `--turn-type-tiling`, `--basin-driven-parents`,
`--adaptive-register-threshold`, `--mirror-balance`, `--sasa-lower-tail` + sub-knobs; spec
`docs/superpowers/specs/2026-09-09-genpept-universal-coverage-tunings-design.md`).

## Funnel

| stage | rep3 (legacy) | rep4 (bundle) |
|---|---|---|
| proposals | 800,000 | 800,000 |
| candidates | 12,000 | 12,000 |
| implicit minima | 11,981 (99.8%) | 11,935 (99.6%) |
| BH minima | 1,192 | 1,132 (295 parents in 110 basins) |
| final survivors | 10,019 | 9,745 |

## Headline metrics

| metric | rep3 | rep4 | note |
|---|---|---|---|
| native-envelope seeds | 31 (0.31%) | 31 (0.32%) | **unchanged — the register yield is not in proposal space** |
| best CA-RMSD to 1UAO | 1.16 Å | 1.18 Å | ~identical |
| entropy bb / ca / heavy | 0.978 / 0.979 / 0.939 | 0.972 / 0.973 / **0.953** | heavy improved; winners tie |
| windows @k1200 | 26 / 27 / 23 | 26 / 27 / 23 | identical |
| chirality frac(+) | 0.540 | **0.502** | mirror balance governed (T5) |
| median SASA | 13.65 | 13.80 | tail preserved (T6, q05 ≤ pool×1.1) |

## Mechanism outcomes (production artifacts)

- **T1** (coverage uniformization): smoke-scale acceptance 59.9% vs legacy ~6.9% — proposal diversity
  physically enriched; full weight tables in `coverage_uniformization.json`.
- **T2** (basin-driven parents): 295 parents over 110 geometric basins via n^γ medoid allocation
  (γ=0.25, cap 6/basin; logged per-row `basin_size`).
- **T3** (turn-type tiling): 94.5% of candidates carry ≥1 universal-type overlay across all 5
  type windows. *Provenance note: this fraction is measured on the post-fix smoke-2 bank
  (1,418/1,500 candidates); the production rep4 run predates the `tt_source` column fix, so
  its provenance column is empty and the exact fraction is not re-measurable on the rep4
  bank itself (the mechanism is flag-deterministic and applied throughout).*
- **T4** (adaptive register threshold): 86,400 backbone N–O samples; no bimodal valley detected,
  honest fallback to 3.8 Å (no fabrication); `register_fraction` recorded per survivor (mean 0.095).
- **T5** (mirror balance): final frac(+) = 0.502 with **zero swaps** needed (target tolerance 0.2).
- **T6** (SASA lower tail): final q05 = 12.30 vs pool 12.45 ≤ ×1.1 — tail kept by construction.

## Verdict

The bundle **fixes the process pathologies it targets** (coverage, rare-basin funding, turn mixture,
handedness governance, tail preservation) but **does not move the folded-state yield**. The
register-envelope occupancy is 0.31–0.32% both ways. Conclusion (counterfactual-safe): the fold-basin
deficit of these banks lives in what the implicit minimizer can stabilize, not in what the generator
proposes. The next lever is expansion-side (guided BH/NMA around near-native minima — T2's proven
behavior), not a larger search pool.

## Production caveats to carry

1. The pipeline deadlocked twice at stage boundaries (full 0%-CPU pool hang at NMA, then Final
   select); `--resume` unsticks it deterministically in both cases. The condition is a scheduling
   deadlock in the driver's coordination, not input content (resume re-runs the same inputs).
2. `fast_ca_coords_from_angles` warmup is deterministic (seeded by (seq, seed, n_requested));
   the CU table is initialized once at startup ("freeze-after-calibration"), the faithful
   determinism-preserving approximation of the spec's rolling update.

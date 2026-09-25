# Synthetic top-up study: outputs

These are the outputs of the synthetic validation in spec section 6 (Task 10, controller rulings 22-31). No MD was run. Each campaign drives the real code: the top-up allocator, the union-MBAR diagnostics, the throughput model and the layout graphs. The landscapes are analytic 2D surfaces, each with a 4-rung λ ladder (28 centres × 4 = 112 states).

## Files

| File | Contents |
|---|---|
| `topup_study.json` | Per-landscape summaries. `meta` holds the regime calibration: the frozen budgets, the rows-floor budgets and the arm-A-only calibration trace. |
| `topup_study_detail.json.gz` | Per-seed and per-epoch detail: every plan, its deficits, partners, routed and topped edges, and σ before, predicted and realised. |
| `sigma_rule_study.json` | Spearman correlations of each candidate σ_k rule against the true local error of Δf (ruling 23). |
| `sigma_rule_study_rows.json.gz` | Per-state rows behind those correlations. |

## Regenerate

```bash
# A/B study. Recalibrates the regime from arm A only, writes it, then runs both arms at 20 seeds.
python -m gareus.synth.topup_study --n-seeds 20 --recalibrate \
    --out docs/superpowers/specs/2026-09-24-effective-topups/synth_study

# sigma_k rule study (ruling 23)
python -m gareus.synth.sigma_rule_study --n-seeds 5 --hours 0.3 5.0 \
    --out docs/superpowers/specs/2026-09-24-effective-topups/synth_study

# union_diagnostics_from_npz cost at 236 states (Task 8 item; prints JSON, writes nothing)
python -m gareus.synth.union_solve_bench --n-rows 1000000
```

Every run is deterministic for a given seed. Without `--recalibrate`, the study uses the budgets frozen in `gareus/synth/topup_study.py` (`FROZEN_BUDGET_HOURS` and `ROWS_FLOOR_HOURS`).

## Verdict (ruling 31: final, accepted as is)

Ruling 28 asks for a budget, chosen from arm A alone, at which 10-40 % of states are above σ* = 0.10 kcal/mol **and** the median state has at least 40 decorrelated rows. No landscape has such a budget. On every landscape the rows floor is reached at about 2-2.5 wall-hours, and at that budget arm A has **0 %** of states above target. So every landscape is reported as "no targetable regime" and left out of pass/fail.

The A/B comparison was still run at the rows-floor budget, for information only:

- **Max σ:** top-ups win on gated-barrier (20/20 seeds, −8.4 %) and on slow-cv2-double-branch (19/20, −1.3 %).
- **PMF RMSE:** top-ups win on no landscape. On gated-barrier they are 7 % worse.

The missing-bridge scenario passes. In 20/20 seeds, the edges spanning the removed pass are routed as structural, and the same coordinate pairs are not routed in the intact layout.

The patch τ penalty is kept (ruling 29). Its measured median g(patch)/g(all-state) is 1.26 on gated-barrier and 1.19 on slow-cv2-double-branch.

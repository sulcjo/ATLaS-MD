# CV1 steerability + bank axis stability (+ eigenvalue-gap robustness)

> **Provenance caveat (2026-09-10 audit):** the steerability table and the half-bank column
> of the stability table below were produced by a fuller script version than the one shipped
> in `scripts/`; the shipped `genpept_r7_steer_stability.json` persists only the plain-PC1
> (median 0.993, min 0.975 — matches row 1) and residual-PC1 (median 0.960, min 0.561)
> half-bank statistics plus the generator-split columns (which match exactly). The plain-PC2
> and residual-combined half-bank cells have no persisted backing; the residual-vs-bb row's
> "0.90 (0.02–0.56)" differs from the JSON's res1 (0.960/0.561). Treat those cells as
> console-transcribed, not artifact-backed.

## Steerability (r7 bank; RMS |∂CV/∂x| per atom, /nm/term, measured in each CV's own p05–p95 band)

| def | pairs | RMS |∇CV| in band | frac of bank with ~zero gradient |
|---|---|---|---|---|
| bb r≥4, r₀=10 Å, β=3 | 336 | 4.3e-4 | 0.0% |
| ca r≥4, r₀=10 Å, β=3 | 21 | 8.1e-4 | 0.0% |
| heavy r≥4, r₀=12 Å, β=3 | 1256 | 2.2e-4 | 0.0% |
| legacy heavy r₀=4.5, β=6 | 1256 | 0.0e0 (unresolvable) | **7.7%** of bank has ~zero gradient *even inside its own 5–95% band* |

Reading: all three winner defs keep a non-vanishing drive over their whole useful range;
the CA-only variant has the largest per-term sensitivity (fewest terms); the legacy def
has a genuine gradient dead-zone, matching the pull-stall failure measured on production.

## Axis stability across resamples / generator subsets

| axis | half-bank 30× |cos| median (min) | initial-generator (n=1,014) | NMA-generator (n=793) |
|---|---|---|---|---|
| plain torsion PC1 | 0.994 (0.975) | 0.979 | 0.964 |
| **plain torsion PC2** | **0.983 (0.944)** | **0.965** | **0.856** |
| residual-vs-bb PC1 | 0.90 (0.02–0.56) | 0.752 | 0.887 |
| residual combined component=2 | 0.746 (0.02) | 0.568 | 0.857 |

Reading: the plain top two PCs are bank-robust directions;
the residual PC1 is **eigenvalue-degenerate** (explained variance 0.101 vs 0.093 for PC1/PC2),
so its scalar direction swings within a near-eigenspace under resampling —
valid as a *subspace*, fragile as a single scalar CV.
Choosing plain torsion PC2 as the CV2 scalar gains both orthogonality-by-construction
(see orthogonal-pairs table: |r_s| 0.04–0.07 vs every candidate CV1) and demonstrated axis robustness.

Data: `genpept_r7_steer_stability.json`; figure: `figures/cv_steerability.png`.

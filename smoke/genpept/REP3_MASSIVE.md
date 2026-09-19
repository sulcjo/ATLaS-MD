# rep3_massive — ≥10,000-seed bank, scaled r7 parameters

Config: `GENPEPT_R7_CV_CENSUS_REPORT/scripts/chignolin_genpept_rep3_massive.yaml` (seed 777001).
Scaled from r7's exact record: n=800k proposals, n_candidate=12,000, n_final=10,500,
BH parents 300×(3+1), NMA 2 modes × {0.75,1.35 Å} × 2 signs, explore 5 rounds keep 800.
Output: `chignolin_genpept_rep3_massive/` (~1.5 GB). Wall time ≈ 2.5 h on the box's RTX 3060 Ti (OpenCL, jobs 16, min_jobs 1).

## Pipeline outcome (per GENPEPT_turbo_summary.json)

| stage | r7 | rep3_massive |
|---|---|---|
| proposals | 200,000 | 800,000 |
| candidates | 3,000 | 12,000 |
| implicit minima | 2,994 (99.8%) | **11,981 (99.8%)** |
| BH minima | 480 | 1,192 |
| NMA minima | 3,840 | 9,536 |
| final survivors | 1,970 | **10,019** |

## Champion-CV replication (bb r≥4 r₀=10 β=3)

- median 0.535 (r7: 0.481), IQR 0.386 (0.393), entropy **0.978 (0.980)**, nw@k1200 **26 (26)**
- plain torsion PCA EVR[1-5]: [0.111, 0.089, 0.078, 0.076, 0.068]
  (r7: [0.122, 0.092, 0.076, 0.074, 0.072]; rep2: [0.111, 0.096, 0.084, 0.077, 0.074])
- contact biasing reproduces the funnel (candidate contact_count mean 8.02).

Verdict: the r7 campaign's statistical conclusions hold on a 5× larger bank;
the CV1 winner and window-count ceilings are bank-size stable.

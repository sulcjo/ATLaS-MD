# rep4: from search density to a thermodynamic starting FES

Bank: 9745 survivors; T = 300 K (kT = 2.494 kcal/mol = 10.4 kJ/mol).
CU table coverage: 100.0% of survivors fall inside the frozen
(Rg, e2e, contact-count) weight table (weight 1.0 assumed outside).

## 1. Three weightings of the same bank (figure cv_rep4_thermo_fes.png)

- raw vs de-biased proposal map: Spearman 0.893 — the CU flattening is visible but modest
- raw vs thermodynamic map: Spearman 0.675 — **Boltzmann re-weighting reshapes the FES substantially**
- proposal vs thermodynamic: Spearman 0.708

## 2. F(Rg) shift (figure cv_rep4_thermo_basins.png)

- compact bins (Rg < 6 A): mean dF = +14.4 kcal/mol
- mid (6-7 A): +41.5; extended (> 7 A): +69.4

## 3. Basin multiplicity thermodynamics

- 9745 basins over 22378 viable minima (implicit + NMA), median g = 2
- multiplicity term kT·ln g: median 1.7,
p95 4.5 kcal/mol — comparable to whole-bank energy spread
- top-20 basins by mean-E vs by F: overlap 11/20 (by min-E: 11/20)
— small, heavily-populated basins win under F where deep, lonely ones win under E

## Caveats

- E is a GBn2 minimized energy: solvent-model-dependent (see consensus check), no explicit counterions
- multiplicity g counts *viable minimization outcomes*, a proxy for basin volume, not a true configurational integral
- w_CU is the frozen startup table; the spec's rolling update was approximated (freeze-after-calibration)
- this is a *starting* FES for window initialization / bias sanity, not a converged PMF

# rep4: consensus implicit-solvent model check

All 9745 survivors re-scored under GBn2 (anchor vs stored minimized energy:
max |diff| = 0.0 kcal/mol) and OBC2 single points.

- cross-model agreement: Spearman(E) = **0.986**;
model gap E_OBC2 - E_GBn2 std = 1.0 kcal/mol
- curl preference (rho flank-psi vs E): GBn2 +0.121 vs OBC2 +0.143
- compaction preference (rho CV1 vs E): GBn2 -0.182 vs OBC2 -0.153
- most-stable-quintile overlap: 18.3% of survivors are stable
in both models; 1.7% stable only under GBn2;
1.7% only under OBC2

Reading: where the two solvent models disagree, single-model energy claims
(survivor ranking, curl bias, basin depths in THERMO_STARTING_FES.md) carry a
model systematic of the size of the disagreement. Treat consensus-unstable
survivors as solvent-model-sensitive.

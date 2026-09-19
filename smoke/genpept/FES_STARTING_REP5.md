# rep5 (T7): starting FES — heavy contacts (r0=12, b3) x Rg

9886 rep5 survivors vs 9745 rep4; T = 300 K;
weights = w_CU^-1 x exp(-(E-E_min)/kT) (each bank's own CU table + GBn2 energies).

- rep5 raw-vs-thermo map agreement: Spearman 0.717
- rep5 thermo minimum at heavy CV1 = 0.77, Rg = 5.69 A
- map difference (rep5 - rep4): -15.1 .. +11.8 kcal/mol
- native-envelope seeds (n=39): median Rg 5.76 A,
  median heavy CV1 0.84; lowest thermo F on their bins 4.13 kcal/mol

Figures: cv_rep5_fes_heavy_rg.png (3 maps), cv_rep5_fes_vs_rep4.png (1-D + dF map).

Caveats: GBn2 single-solvent systematic ~1 kcal; not a converged PMF; the T7 A/B
carries a small GPU-noise floor at the BH stage (see REP5_T7.md).

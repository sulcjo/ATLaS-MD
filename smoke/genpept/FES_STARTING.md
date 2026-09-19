# rep4: the starting FES on the working CV axes

9745 survivors, T = 300 K. Weights: w = w_CU^-1 (recorded acceptance table) x
exp(-(E-E_min)/kT) (GBn2 minimized energies, matched by basename).
Maps on CV1 (bb r0=10 b3) x plain torsion PC1, 45x45 grid, red stars = native-envelope
seeds (n=31). Figures: cv_rep4_fes_starting.png (3 maps),
cv_rep4_fes_starting_1d.png (1-D profile + delta).

- map agreement raw vs thermo (CV1 x PC1): Spearman 0.526
- 1-D along CV1: thermo shifts bins by -0.3 to +19.6 kcal/mol
- landscape depth: raw 1.7 vs thermo 21.0 kcal/mol

Usage: initialize GaREUS window centers/widths from the thermo map, use raw map
only as a coverage diagnostic. Caveats: GBn2 single-solvent systematic ~1 kcal
(consensus check), multiplicity not yet folded in per-seed (see THERMO_STARTING_FES.md
for the basin-level version), not a converged PMF.

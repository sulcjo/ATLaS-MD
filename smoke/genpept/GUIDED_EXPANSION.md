# Guided register-rewarded BH expansion (demo on rep4 near-native tail)

30 best-RMSD rep4 survivors; 8 BH steps each (675 K MD kick 350 x 0.0035 ps
+ 300-iter GBn2 minimization); Metropolis on F = E - lambda * n_register
(backbone N-O < 3.8 A, |i-j| >= 3); lambda = 0 (control) vs 1.5 kcal per register.

- control: mean RMSD 1.57 -> 1.98 A,
  mean register count 2.0 -> 2.0,
  envelope seeds 4/30
- rewarded: mean RMSD 1.57 -> 1.98 A,
  mean register count 2.0 -> 2.0,
  envelope seeds 4/30

Figure: cv_rep4_guided_expansion.png. This is the expansion-side lever the rep4 verdict
pointed at, tested without touching GENPEPT.py.

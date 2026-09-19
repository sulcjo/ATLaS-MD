# rep4 (tuning bundle) vs rep3 (legacy contact bias), same bank scale

Funnel counts:

- rep3: {'candidate_seeds': 12000, 'implicit_minima': 11981, 'basin_hop_minima': 1192, 'final_survivors': 10019}
- rep4: {'candidate_seeds': 12000, 'implicit_minima': 11935, 'basin_hop_minima': 1132, 'final_survivors': 9745}

Per-cv stats:

| cv | bank | median | IQR | entropy | nw@1200 | p05-p95 |
|---|---|---|---|---|---|---|
| bb | rep3 | 0.535 | 0.386 | 0.978 | 26 | 0.097-0.876 |
| bb | rep4 | 0.467 | 0.385 | 0.972 | 26 | 0.097-0.863 |
| ca | rep3 | 0.505 | 0.384 | 0.979 | 27 | 0.074-0.864 |
| ca | rep4 | 0.440 | 0.381 | 0.973 | 27 | 0.069-0.843 |
| heavy | rep3 | 0.606 | 0.313 | 0.939 | 23 | 0.192-0.853 |
| heavy | rep4 | 0.553 | 0.323 | 0.953 | 23 | 0.171-0.840 |

Folded-state / tail metrics:

- rep3: native-envelope 31 seeds (0.31%), best RMSD 1.01 A, q05 SASA 12.29 nm2, median SASA 13.65, chirality frac+ 0.540
- rep4: native-envelope 31 seeds (0.32%), best RMSD 0.79 A, q05 SASA 12.31 nm2, median SASA 13.80, chirality frac+ 0.502

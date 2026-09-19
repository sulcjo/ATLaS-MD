# rep4 (tuning bundle) folded-state proximity audit + pseudo-FES maps

Reference: PDB 1UAO (chignolin GYDPETGTWG NMR ensemble, 18 models). Native pair distances: d(D3 N, G7 O) 6.17..7.56 A; d(D3 N, T8 O) 2.84..4.22 A.


## rep4: folded-proximity audit (vs 1UAO)

- min CA-RMSD to any 1UAO model: **0.79 A** (seed `survivor_6532_basin_6885_candidate_4445_gencluster_416_conf_139572_implicit_min`, model 9)
  - CA-RMSD < 2.0 A: 0.84% (82 seeds)
  - CA-RMSD < 2.5 A: 5.45% (531 seeds)
  - CA-RMSD < 3.0 A: 19.81% (1930 seeds)
  - CA-RMSD < 3.5 A: 39.84% (3882 seeds)
  - CA-RMSD < 4.0 A: 59.44% (5792 seeds)
- median: 3.75 A
- folded corner (both d37, d38 < 4 A): **0 seeds**; both < 5 A: 8
- minima: d37 min 2.98 A, d38 min 2.84 A
- DSSP: mean strand (E) residues/seed 0.04; helix (H) 0.98

## rep4 pseudo-FES
- grid occupancy on CV1xPC1 map: 63.3%
- figures: cv_rep4_pseudofes.png (3 maps), cv_rep4_folded_proximity.png

# Folded-state proximity audit + rep3 pseudo-FES maps

Reference: PDB 1UAO (chignolin GYDPETGTWG NMR ensemble, 18 models, fetched from RCSB). Native pair distances over the ensemble: d(D3 N, G7 O) 6.17..7.56 A; d(D3 N, T8 O) 2.84..4.22 A.


## r7: folded-proximity audit (vs 1UAO)

- min CA-RMSD to any 1UAO model: **1.61 A** (seed `survivor_1120_basin_1138_candidate_1316_gencluster_1316_conf_026261_implicit_min`, model 11)
  - CA-RMSD < 2.0 A: 0.71% (14 seeds)
  - CA-RMSD < 2.5 A: 6.95% (137 seeds)
  - CA-RMSD < 3.0 A: 22.39% (441 seeds)
  - CA-RMSD < 3.5 A: 44.31% (873 seeds)
  - CA-RMSD < 4.0 A: 61.73% (1216 seeds)
- median: 3.63 A
- folded corner (both d37, d38 < 4 A): **0 seeds**; both < 5 A: 0
- minima: d37 min 2.96 A, d38 min 4.50 A
- DSSP: mean strand (E) residues/seed 0.01; helix (H) 0.91

## rep3_massive: folded-proximity audit (vs 1UAO)

- min CA-RMSD to any 1UAO model: **1.01 A** (seed `survivor_1518_basin_1561_nma_bh_candidate_2172_gencluster_2172_conf_351390_implicit_min_hop_000_mode01_p0.75A_implicit_min`, model 15)
  - CA-RMSD < 2.0 A: 1.20% (120 seeds)
  - CA-RMSD < 2.5 A: 7.93% (795 seeds)
  - CA-RMSD < 3.0 A: 25.67% (2572 seeds)
  - CA-RMSD < 3.5 A: 48.16% (4825 seeds)
  - CA-RMSD < 4.0 A: 66.03% (6616 seeds)
- median: 3.55 A
- folded corner (both d37, d38 < 4 A): **0 seeds**; both < 5 A: 20
- minima: d37 min 2.85 A, d38 min 2.87 A
- DSSP: mean strand (E) residues/seed 0.02; helix (H) 0.82

## rep3 pseudo-FES
- grid occupancy on CV1xPC1 map: 68.9%
- figures: cv_rep3massive_pseudofes.png (3 maps), cv_rep3_massive_folded_proximity.png

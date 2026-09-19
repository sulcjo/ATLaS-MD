# Solvation as a CV axis (rep3 bank)

## Orthogonality vs everything else (Spearman; top |rho| per solvation axis)
- gb_solvation: e2e_nc_ca -0.27; rg_heavy -0.25; c_bb_s4_r010_b3 +0.20; hairpin_closure_ratio +0.19; MI vs CV1 0.086, vs PC2 0.017; joint occ/cond-IQR vs CV1 1.00/0.95
- sasa_total: rg_heavy +0.91; c_bb_s4_r010_b3 -0.80; shape_anisotropy +0.63; e2e_nc_ca +0.57; MI vs CV1 0.546, vs PC2 0.020; joint occ/cond-IQR vs CV1 0.89/0.51
- sasa_hydrophobic_frac: hairpin_closure_ratio +0.26; c_bb_s4_r010_b3 +0.24; torpca_pc1 -0.19; e2e_nc_ca -0.18; MI vs CV1 0.065, vs PC2 0.011; joint occ/cond-IQR vs CV1 1.00/0.97
- burial_y2: rg_heavy -0.56; c_bb_s4_r010_b3 +0.40; shape_anisotropy -0.38; torpca_pc2 +0.29; MI vs CV1 0.101, vs PC2 0.084; joint occ/cond-IQR vs CV1 0.99/0.86
- burial_w9: rg_heavy -0.53; shape_anisotropy -0.38; hairpin_closure_ratio -0.30; c_bb_s4_r010_b3 +0.28; MI vs CV1 0.051, vs PC2 0.015; joint occ/cond-IQR vs CV1 1.00/0.96
- burial_d3: torpca_pc2 -0.60; c_bb_s4_r010_b3 +0.22; shape_anisotropy -0.21; rg_heavy -0.19; MI vs CV1 0.055, vs PC2 0.246; joint occ/cond-IQR vs CV1 1.00/1.00

## Folded-state relation
- 1UAO gb_solvation: -1553 kJ/mol [-1782,-1414]; bank fraction with gb <= native: 0.5255
- 1UAO SASA mean 11.2 nm^2 vs bank median 13.6; bank fraction <= native: 0.0003
- near-native envelope (n=31): gb mean -1525.0 vs rest -1554.6 (delta +29.7 kJ/mol, permutation p~0.18 previously)

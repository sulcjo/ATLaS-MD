# Mutual information, discrete intermediate states, full-bank energy decomposition

## MI (Kraskov k=3, nats) of c_bb_s4_r010_b3 with CV2 candidates
- restorpca_pc1: MI 0.089 (null -0.020), excess +0.109, rho +0.096
- torpca_pc2: MI 0.061 (null -0.008), excess +0.069, rho -0.050
- torpca_pc1: MI 0.263 (null -0.003), excess +0.266, rho +0.501
- acylindricity: MI 0.016 (null -0.012), excess +0.028, rho +0.069
- hairpin_closure_ratio: MI 0.102 (null -0.017), excess +0.119, rho +0.254
- chirality_ca: MI 0.044 (null -0.014), excess +0.058, rho +0.059
- pack_y2: MI -0.163 (null -0.189), excess +0.026, rho +0.277
- burial_d3: MI 0.046 (null -0.016), excess +0.062, rho +0.161
- turn3537_mean_ca: MI 0.315 (null -0.019), excess +0.334, rho -0.537
- contact_order_ca8: MI 0.967 (null -0.048), excess +1.016, rho +0.933

## KDE states along residual CV2
- S0 [-2.87,-1.70] n=121 (6.1%), rep=survivor_1827_basin_1856_candidate_1024_gencluster_1024_conf_011514_implicit_min, psi means {'psi Y2': 148.8120574951172, 'psi D3': -40.66843032836914, 'psi P4': 144.91773986816406, 'psi G7': 43.99459457397461, 'psi T8': 135.98419189453125}
- S1 [-1.70,-1.28] n=137 (7.0%), rep=survivor_1815_basin_1844_nma_bh_candidate_369_gencluster_369_conf_030202_implicit_min_hop_002_mode02_m1.35A_implicit_min, psi means {'psi Y2': 141.60968017578125, 'psi D3': -21.826812744140625, 'psi P4': 142.4442901611328, 'psi G7': 57.93585968017578, 'psi T8': 61.674747467041016}
- S2 [-1.28,-0.62] n=312 (15.8%), rep=survivor_853_basin_867_nma_bh_candidate_2663_gencluster_717_conf_014249_implicit_min_hop_002_mode02_m0.75A_implicit_min, psi means {'psi Y2': 125.72746276855469, 'psi D3': -15.423497200012207, 'psi P4': 136.04013061523438, 'psi G7': 31.023496627807617, 'psi T8': 76.4662094116211}
- S3 [-0.62,0.18] n=494 (25.1%), rep=survivor_558_basin_568_explore_r03_hit_0066_conf_017941_implicit_min, psi means {'psi Y2': 90.58314514160156, 'psi D3': 13.763384819030762, 'psi P4': 73.81634521484375, 'psi G7': 36.74046325683594, 'psi T8': 48.629295349121094}
- S4 [0.18,1.02] n=544 (27.6%), rep=survivor_998_basin_1015_candidate_845_gencluster_845_conf_072240_implicit_min, psi means {'psi Y2': 11.089923858642578, 'psi D3': -10.627289772033691, 'psi P4': 8.85440731048584, 'psi G7': 18.71012306213379, 'psi T8': 9.567401885986328}
- S5 [1.02,2.54] n=362 (18.4%), rep=survivor_810_basin_823_nma_bh_candidate_1519_gencluster_1519_conf_015687_implicit_min_hop_001_mode02_p1.35A_implicit_min, psi means {'psi Y2': -15.464920043945312, 'psi D3': -38.243011474609375, 'psi P4': -7.967069625854492, 'psi G7': -5.003448009490967, 'psi T8': -7.711217403411865}

## GBn2 component stats (1,970 seeds)
- bond: entropy 0.841, BC 0.346, rho_cv1 +0.23, rho_cv2 +0.21, std 1.5, p5–p95 [16,21]
- angle: entropy 0.811, BC 0.349, rho_cv1 +0.33, rho_cv2 +0.31, std 9.2, p5–p95 [59,89]
- torsion: entropy 0.727, BC 0.323, rho_cv1 +0.16, rho_cv2 -0.02, std 12.9, p5–p95 [351,392]
- nonbonded: entropy 0.845, BC 0.331, rho_cv1 -0.31, rho_cv2 -0.12, std 129.1, p5–p95 [-720,-294]
- gb_solvation: entropy 0.854, BC 0.322, rho_cv1 +0.23, rho_cv2 +0.04, std 116.5, p5–p95 [-1746,-1359]
- total: entropy 0.805, BC 0.296, rho_cv1 -0.34, rho_cv2 -0.31, std 24.0, p5–p95 [-1636,-1558]

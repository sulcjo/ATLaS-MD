# Rigor & hand-off pack

## Bootstrap 95% intervals (400 resamples)
- c_bb_s4_r010_b3: entropy 0.979 [0.974,0.986]; nw 26 [26,27]
- c_ca_s4_r010_b3: entropy 0.976 [0.970,0.981]; nw 27 [26,28]
- c_heavy_s4_r012_b3: entropy 0.957 [0.949,0.965]; nw 24 [24,24]
- cresb_heavy_s4_r012_b3: entropy 0.956 [0.947,0.961]; nw 24 [23,24]
- torpca_pc1: entropy 0.928 [0.920,0.937]; nw 140 [135,146]
- restorpca_pc1: entropy 0.915 [0.908,0.938]; nw 129 [127,133]
- trace_own_pc1_C1: median 0.388 [0.269,0.552]

## Morph path realism
- geodesic 73->1259: 9 steps, 50% edges within median kNN dist (2.87); linear interpolation: 86% of 21 points within, max gap 3.15
- geodesic PDB: figures/r7v2_residual_cv2_geodesic_path.pdb (9 models)

## Provenance stratification (survivor bank)
- initial (n=1014): CV1 0.429±0.235, CV2 -0.024, frac(cv2>1) 0.18, frac(cv2<-1) 0.19
- BH (n=3): CV1 0.676±0.075, CV2 +0.241, frac(cv2>1) 0.33, frac(cv2<-1) 0.00
- NMA (n=793): CV1 0.478±0.224, CV2 +0.013, frac(cv2>1) 0.21, frac(cv2<-1) 0.19
- PCA-frontier (n=160): CV1 0.734±0.192, CV2 +0.087, frac(cv2>1) 0.16, frac(cv2<-1) 0.14

## X-Pro / X-Y omega cis fractions
- omega D3-P4: 0.0030
- omega P4-E5: 0.0000

## Contact-bias audit (accepted candidates vs reconstructed proposal space; strength 0.15, coded ref = 21)
- candidate pool (n=3,000) contact_count mean 7.97; final survivors (n=1,970) 7.64
- inverse-acceptance-weighted reconstruction of the UNBIASED proposal space: mean 4.79
- enrichment of compact states by the bias: frac(cc>=5) 0.679 vs 0.417; cc>=10 0.371 vs 0.133;
  cc>=15 0.128 vs 0.026; cc>=20 0.020 vs 0.002
- bank-side coverage of compact conformers is therefore enriched ~3-5x over unguided generation.

## Ladder tiling
- k-center (z-scored CV1,CV2,Rg,e2e) K=16 seeds written to seed_bank_tiling_k16.csv

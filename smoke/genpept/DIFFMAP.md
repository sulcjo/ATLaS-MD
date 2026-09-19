# Diffusion-map / intrinsic-dimension analysis of the r7 seed bank

bank: 1970 seeds; metrics: z-scored interleaved torsion sin/cos (36-D) and z-scored CA dRMSD (45-D).

## TwoNN intrinsic dimension
- torsion space: d = 3.60 ± 0.05
- CA-dRMSD space: d = 4.64 ± 0.03

## Spectra (lambda_k, gap; k=1 is the trivial psi0)
- torsion: k=1:1.000(-0.998), k=2:0.002(-0.001), k=3:0.002(-0.000), k=4:0.001(-0.000), k=5:0.001(-0.000), k=6:0.001(-0.000), k=7:0.001(-0.000)
- ca_drmsd: k=1:1.000(-0.993), k=2:0.007(-0.004), k=3:0.003(-0.001), k=4:0.002(-0.001), k=5:0.002(-0.000), k=6:0.001(-0.000), k=7:0.001(-0.000)

## Epsilon robustness of psi1
- torsion: |rho| vs 0.5x eps = 1.000, vs 2.0x eps = 1.000
- ca_drmsd: |rho| vs 0.5x eps = 1.000, vs 2.0x eps = 1.000

## Eigenvector identification (Spearman top |r| vs census CVs)
- torsion psi1: torpca_pc1 +0.964, c_ca_s2_r06_b6 +0.815, c_bb_s2_r04.5_b6 +0.811, turn3537_mean_ca -0.807
- torsion psi2: torpca_pc2 -0.928, restorpca_pc2 -0.765, burial_d3 -0.606, restorpca_pc1 -0.513
- torsion psi3: torpca_pc3 -0.735, restorpca_pc3 -0.734, torpca_pc4 +0.367, d_e5_t8_ca +0.328
- torsion psi4: torpca_pc5 -0.486, torpca_pc4 -0.428, d_e5_t8_ca +0.367, hbond_crossstrand -0.323
- torsion psi5: torpca_pc4 -0.527, torpca_pc5 +0.483, torpca_pc3 -0.344, restorpca_pc3 -0.335
- torsion psi6: torpca_pc5 -0.428, restorpca_pc3 -0.385, torpca_pc3 -0.372, torpca_pc4 -0.318
- torsion psi7: d_p4_g7_ca -0.307, d_e5_t8_ca +0.251, hbond_crossstrand +0.223, pack_y2 +0.181
- torsion psi8: torpca_pc5 -0.247, turn3537_mean_ca -0.212, pack_w9 +0.205, c_sch_s2_r08_b6 +0.197
- torsion psi9: rama_ppii +0.324, rama_left -0.282, turn3537_mean_ca +0.193, torpca_pc4 +0.182
- torsion psi10: rama_left +0.275, rama_ppii -0.271, torpca_pc5 -0.239, torpca_pc4 +0.233
- torsion psi11: rama_left -0.281, d_e5_t8_ca +0.140, d_d3_t8_ca +0.122, rama_alpha +0.116
- torsion psi12: torpca_pc4 +0.189, d_e5_t8_ca +0.176, d_d3_t8_ca +0.129, c_sca_s2_r08_b6 -0.123
- ca_drmsd psi1: mean_ca_ca -0.983, rg_ca -0.975, c_ca_s2_r010_b1.5 +0.969, c_bb_s2_r010_b1.5 +0.963
- ca_drmsd psi2: torpca_pc5 -0.489, d_e5_t8_ca +0.333, burial_y2 +0.286, pack_y2 +0.240
- ca_drmsd psi3: restorpca_pc1 -0.626, torpca_pc1 -0.516, acylindricity +0.457, c_bb_s2_r04.5_b6 -0.436
- ca_drmsd psi4: d_e5_t8_ca +0.559, hairpin_closure_ratio +0.489, d_p4_g7_ca +0.468, e2e_nc_ca -0.462
- ca_drmsd psi5: torpca_pc4 -0.459, d_e5_t8_ca -0.409, d_p4_g7_ca +0.393, hbond_crossstrand -0.274
- ca_drmsd psi6: torpca_pc5 -0.568, restorpca_pc2 +0.344, torpca_pc3 +0.340, restorpca_pc3 +0.327
- ca_drmsd psi7: restorpca_pc3 +0.697, torpca_pc3 +0.692, burial_d3 -0.550, burial_y2 +0.441
- ca_drmsd psi8: torpca_pc2 -0.706, restorpca_pc2 -0.541, restorpca_pc1 -0.459, burial_d3 -0.351
- ca_drmsd psi9: torpca_pc3 -0.302, restorpca_pc3 -0.302, burial_w9 +0.268, pack_w9 +0.245
- ca_drmsd psi10: torpca_pc4 +0.649, pack_w9 +0.280, burial_w9 +0.257, burial_d3 +0.191
- ca_drmsd psi11: c_ca_s3_r06_b6 +0.305, c_ca_s3_r08_b3 +0.299, c_ca_s2_r08_b3 +0.298, c_ca_s3_r08_b6 +0.298
- ca_drmsd psi12: d_d3_t8_ca +0.246, pack_p4 -0.198, restorpca_pc2 +0.184, hairpin_closure_ratio +0.183

## LASSO identification (interpretable features)
- torsion_psi1: R2=0.974  torpca_pc1(+0.83), rama_alpha(+0.12), rama_ppii(+0.12), d_p4_g7_ca(-0.08)
- torsion_psi2: R2=0.937  torpca_pc2(-0.76), restorpca_pc2(-0.18), torpca_pc5(-0.18), torpca_pc1(-0.15)
- torsion_psi3: R2=0.826  torpca_pc1(-0.82), torpca_pc3(-0.71), turn3537_mean_ca(-0.51), restorpca_pc1(+0.48)
- ca_drmsd_psi1: R2=0.985  rg_heavy(-0.35), e2e_nc_ca(-0.27), c_ca_s4_r010_b3(+0.21), torpca_pc1(+0.14)
- ca_drmsd_psi2: R2=0.647  c_bb_s4_r010_b3(-0.95), d_y2_w9_ca(+0.90), c_ca_s4_r010_b3(+0.74), turn3537_mean_ca(+0.69)
- ca_drmsd_psi3: R2=0.936  d_y2_w9_ca(-0.70), restorpca_pc1(-0.68), rg_heavy(+0.60), turn3537_mean_ca(+0.53)

psi vectors per seed (first 6 per metric): genpept_r7_diffmap_psi.csv; machine-readable: genpept_r7_diffmap.json.
# CV2 deep dive — bank PCA axes and dynamics content

Bank: 1,970 x 36 interleaved sin/cos features (canonical `gareus.tica` ordering).

- plain PCA EVR[1-6]: [0.122, 0.092, 0.076, 0.074, 0.072, 0.064]
- residual-on-c_bb_s4_r010_b3 EVR[1-6]: [0.098, 0.093, 0.078, 0.076, 0.073, 0.065]
- |cos(plain PC1, res PC1)| = 0.166; top-5 subspace overlap (mean cos^2 of principal angles) = 0.986
- top residual-PC1 loadings: cos psi W9 -0.486, cos psi P4 -0.410, cos psi D3 +0.341, sin psi D3 -0.334, cos psi Y2 -0.254, sin phi G7 +0.251, cos psi E5 +0.219, sin psi W9 +0.205, sin psi E5 -0.165, cos phi W9 -0.154

## Diffusion content (Hess cosine C1) on 32 chignolin_6 epoch-0 traces (4,768 frames each)

- plain_pc1: median C1 0.076, frac>=0.5 0.0% (0/32)
- res_pc1: median C1 0.074, frac>=0.5 0.0% (0/32)
- own_pc1: median C1 0.388, frac>=0.5 40.6% (13/32)
- res_pc1 half-traces: median 0.110, frac>=0.5 10.9% (7/64)

Per-trace corr(bank resPC1 projection, stored secondary_cv): median -0.540 range [-0.817, 0.075]

## CV1 x CV2 joint maps (16 x 8 over q0.1-q99.9)

- bank: c_bb_s4_r010_b3 x bank resPC1: occupancy 96.9%, per-CV1-bin CV2 IQR median 1.389
- bank: c_heavy_s4_r012_b3 x bank resPC1: occupancy 93.8%, per-CV1-bin CV2 IQR median 1.408
- chignolin_6 epoch_000: stored primary_cv x bank resPC1 proj: occupancy 96.9%, per-CV1-bin CV2 IQR median 0.776
- chignolin_6 epoch_000: stored primary_cv x stored secondary_cv: occupancy 100.0%, per-CV1-bin CV2 IQR median 1.443

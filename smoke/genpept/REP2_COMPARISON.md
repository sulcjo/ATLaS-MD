# Bank-to-bank reproducibility: r7 vs rep2 (identical GENPEPT params, different seed)

r7 n=1970, rep2 n=1984 (both 200k proposals, bias 0.15, chignolin preset, backend full, OpenCL; seed 12345 vs 54321)

| CV | med r7 | med r2 | IQR r7 | IQR r2 | ent r7 | ent r2 | nw r7 | nw r2 | KS | p |
|---|---|---|---|---|---|---|---|---|---|---|
| bb_r010_b3 | 0.481 | 0.545 | 0.385 | 0.391 | 0.979 | 0.976 | 26 | 26 | 0.088 | 3.6e-07 |
| bb_r010_b6 | 0.482 | 0.547 | 0.393 | 0.396 | 0.980 | 0.977 | 27 | 27 | 0.088 | 4.3e-07 |
| ca_r010_b3 | 0.452 | 0.515 | 0.378 | 0.396 | 0.976 | 0.976 | 27 | 27 | 0.089 | 3.2e-07 |
| heavy_r012_b3 | 0.579 | 0.618 | 0.308 | 0.312 | 0.957 | 0.945 | 24 | 24 | 0.083 | 2.3e-06 |
| heavy_r012_b1.5 | 0.575 | 0.613 | 0.294 | 0.299 | 0.954 | 0.943 | 23 | 23 | 0.081 | 3.7e-06 |
| legacy_heavy_r04.5_b6 | 0.006 | 0.007 | 0.014 | 0.013 | 0.665 | 0.728 | 1 | 1 | 0.043 | 5.1e-02 |

Axis agreement between banks:
- plain_pc1: |cos| 0.9646
- plain_pc2: |cos| 0.5229
- res_pc1: |cos| 0.4529
- plain_pc1x2_top5_overlap: |cos| 0.8648
- plain EVR[1-6] r7 [0.122, 0.092, 0.076, 0.074, 0.072, 0.064] vs r2 [0.111, 0.096, 0.084, 0.077, 0.074, 0.064]

Curl ladder chords (psi circular means, Y2/D3/T8):
- state 0: r7 {'Y2': 135.0655975341797, 'P4': 152.44082641601562, 'W9': 131.40341186523438} vs r2 {'Y2': 33.60662078857422, 'P4': 154.4832763671875, 'W9': 23.21503448486328}
- state 1: r7 {'Y2': 77.9359359741211, 'P4': 137.909423828125, 'W9': 62.74777603149414} vs r2 {'Y2': 14.589754104614258, 'P4': 140.8399658203125, 'W9': 47.793128967285156}
- state 2: r7 {'Y2': 80.63827514648438, 'P4': 35.444393157958984, 'W9': 58.77960205078125} vs r2 {'Y2': 22.991912841796875, 'P4': 33.317813873291016, 'W9': 21.25506591796875}
- state 3: r7 {'Y2': 17.843778610229492, 'P4': 22.10063934326172, 'W9': 13.606278419494629} vs r2 {'Y2': 45.022640228271484, 'P4': -1.1503276824951172, 'W9': 52.579166412353516}
- state 4: r7 {'Y2': 13.333197593688965, 'P4': 2.206354856491089, 'W9': 12.026676177978516} vs r2 {'Y2': 65.04740905761719, 'P4': -8.080880165100098, 'W9': 43.206241607666016}
- state 5: r7 {'Y2': -4.821788311004639, 'P4': -13.033324241638184, 'W9': -5.582344055175781} vs r2 {'Y2': 55.6041259765625, 'P4': -12.515656471252441, 'W9': 136.6327362060547}

## Common-axis projection (rep2 onto r7's plain-PC2 eigenvector) — the meaningful way to compare states
- corr(bank indicator, common-axis projection): +0.037
- state 0 [-2.52,-0.99]: n r7/r2 329/252; Y2 r7 +135 r2 +131; D3 r7 -42 r2 -43; P4 r7 +152 r2 +153; W9 r7 +131 r2 +128
- state 1 [-0.99,-0.47]: n r7/r2 328/374; Y2 r7 +78 r2 +99; D3 r7 -37 r2 -37; P4 r7 +138 r2 +127; W9 r7 +63 r2 +106
- state 2 [-0.47,-0.02]: n r7/r2 328/301; Y2 r7 +81 r2 +105; D3 r7 -29 r2 -18; P4 r7 +35 r2 +70; W9 r7 +59 r2 +101
- state 3 [-0.02,0.48]: n r7/r2 328/348; Y2 r7 +18 r2 +39; D3 r7 -2 r2 +1; P4 r7 +22 r2 -2; W9 r7 +14 r2 +18
- state 4 [0.48,1.06]: n r7/r2 328/363; Y2 r7 +13 r2 +4; D3 r7 +129 r2 +137; P4 r7 +2 r2 +1; W9 r7 +12 r2 +43
- state 5 [1.06,2.40]: n r7/r2 329/346; Y2 r7 -5 r2 -2; D3 r7 +141 r2 +142; P4 r7 -13 r2 -12; W9 r7 -6 r2 +2

## Own-axis-per-bank state comparison is NOT meaningful (and therefore not used elsewhere)
- plain PC2's direction differs across independent banks (|cos| 0.523), so state labels under own axes do not align.
- plain PC1 DOES transfer across banks (|cos| 0.965): compaction-dominant direction is bank-invariant.
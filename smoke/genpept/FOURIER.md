# Fourier analysis of r7 seed-bank dihedrals

bank: 1970 seeds; 18 backbone torsions (phi, psi per residue).

## A. Circular Fourier harmonics R1..R4 per torsion
R1 = concentration at one direction; R2..R4 high = multimodality/periodic components.
- phi Y2     R1=0.843 R2=0.685 R3=0.548 R4=0.515 dir=-85deg
- phi D3     R1=0.839 R2=0.631 R3=0.496 R4=0.422 dir=-79deg
- phi P4     R1=0.981 R2=0.950 R3=0.922 R4=0.871 dir=-72deg
- phi E5     R1=0.800 R2=0.646 R3=0.532 R4=0.453 dir=-82deg
- phi T6     R1=0.804 R2=0.610 R3=0.475 R4=0.477 dir=-86deg
- phi G7     R1=0.609 R2=0.663 R3=0.634 R4=0.469 dir=-79deg
- phi T8     R1=0.768 R2=0.629 R3=0.555 R4=0.520 dir=-83deg
- phi W9     R1=0.760 R2=0.588 R3=0.433 R4=0.396 dir=-87deg
- phi G10    R1=0.628 R2=0.217 R3=0.475 R4=0.586 dir=-119deg
- psi G1     R1=0.904 R2=0.693 R3=0.505 R4=0.394 dir=+156deg
- psi Y2     R1=0.193 R2=0.640 R3=0.153 R4=0.418 dir=+48deg  <- multimodal
- psi D3     R1=0.131 R2=0.734 R3=0.266 R4=0.599 dir=-24deg  <- multimodal
- psi P4     R1=0.163 R2=0.734 R3=0.183 R4=0.491 dir=+48deg  <- multimodal
- psi E5     R1=0.325 R2=0.671 R3=0.220 R4=0.404 dir=+8deg  <- multimodal
- psi T6     R1=0.342 R2=0.657 R3=0.315 R4=0.388 dir=+20deg  <- multimodal
- psi G7     R1=0.306 R2=0.581 R3=0.283 R4=0.208 dir=+22deg  <- multimodal
- psi T8     R1=0.276 R2=0.645 R3=0.214 R4=0.400 dir=+24deg  <- multimodal
- psi W9     R1=0.238 R2=0.724 R3=0.262 R4=0.496 dir=+37deg  <- multimodal

## B. Winding (hidden periodic signals) along ordered coordinates
unwrap angle vs seeds sorted by coordinate; |turns| threshold = max-over-torsions permutation p99 (N=300).
- torsion-psi1 (null p99 36.32): none significant
- ca_drmsd-psi1 (null p99 38.67): none significant
- contact-CV1 (null p99 35.98): none significant
- res-torsion-PC1 (null p99 36.03): none significant

## C. Fisher-Lee circular-circular correlations (top pairs)
- phi T8 ~ psi T8: rho=-0.347
- phi D3 ~ psi D3: rho=-0.340
- phi W9 ~ psi W9: rho=-0.321
- phi E5 ~ psi E5: rho=-0.310
- phi T6 ~ psi T6: rho=-0.289
- phi Y2 ~ psi Y2: rho=-0.266
- phi G7 ~ psi G7: rho=-0.242
- psi Y2 ~ psi E5: rho=+0.186
- psi E5 ~ psi T6: rho=+0.182
- phi W9 ~ psi T8: rho=+0.178

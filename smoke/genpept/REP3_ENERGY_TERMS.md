# GBn2 energy-term behavior on the rep3 FES maps

bank n=10019; terms: bond/angle/torsion strain, direct nonbonded, GB solvation, total.

## Trend slopes (kJ/mol per CV unit, bootstrap CI95)

[cv1]
- bond: +1.6 [+1.5,+1.7]
- angle: +14.0 [+13.0,+15.1]
- torsion: +8.5 [+7.4,+9.6]
- nonbonded: -139.7 [-149.2,-129.6]
- gb_solvation: +95.3 [+86.4,+105.1]
- total: -20.3 [-22.2,-18.5]

[cv2]
- bond: +0.3 [+0.3,+0.3]
- angle: +2.5 [+2.2,+2.7]
- torsion: +1.3 [+1.0,+1.6]
- nonbonded: -6.6 [-9.7,-4.0]
- gb_solvation: +3.4 [+1.1,+5.7]
- total: +1.0 [+0.5,+1.5]

[rg]
- bond: -0.5 [-0.5,-0.5]
- angle: -4.4 [-4.7,-4.1]
- torsion: -2.6 [-2.9,-2.3]
- nonbonded: +48.0 [+45.9,+50.8]
- gb_solvation: -32.9 [-35.2,-30.3]
- total: +7.6 [+7.2,+8.1]

[e2e]
- bond: -0.1 [-0.1,-0.1]
- angle: -0.5 [-0.5,-0.4]
- torsion: -0.3 [-0.3,-0.2]
- nonbonded: +7.2 [+6.7,+7.7]
- gb_solvation: -6.1 [-6.5,-5.7]
- total: +0.2 [+0.1,+0.3]

## Compensation
- partial-term correlation (nonbonded vs gb_solvation): -0.986
- variance compensation coefficient 1 - Var(total)/sum Var(parts): 0.979

## State composition (state mean - bank median, kJ/mol)
- S0: bond -0.1; angle -1.2; torsion +0.0; nonbonded -4.4; gb_solvation +5.1; total -1.3
- S1: bond +0.2; angle +2.6; torsion +2.9; nonbonded -5.9; gb_solvation +0.8; total -0.1
- S2: bond +0.5; angle +3.4; torsion +1.9; nonbonded -25.7; gb_solvation +21.4; total +0.9

## Near-native (native envelope) energy signature vs rest
- n=31; bond +1.1 (p=0.002); angle +5.9 (p=0.012); torsion +5.8 (p=0.036); nonbonded -45.1 (p=0.064); gb_solvation +29.7 (p=0.178); total -2.6 (p=0.581)

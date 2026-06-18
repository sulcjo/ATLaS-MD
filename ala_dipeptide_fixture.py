#!/usr/bin/env python3
"""Emit the canonical capped alanine dipeptide (Ace-Ala-Nme) fixture PDB.

The *.pdb output is gitignored, so this generator is the committed source of
truth. Atom/residue names match amber14-all.xml templates; verified locally to
parameterize with amber14-all.xml + amber14/tip3p.xml (22 particles, 21 bonds).
"""
from __future__ import annotations
import sys
from pathlib import Path

ACE_ALA_NME_PDB = """\
REMARK   ACE-ALA-NME, canonical Amber/OpenMM cap atom names
HETATM    1 HH31 ACE A   1       2.024   1.055   0.337  1.00  0.00           H
HETATM    2  CH3 ACE A   1       2.000   2.090   0.000  1.00  0.00           C
HETATM    3 HH32 ACE A   1       1.378   2.677   0.672  1.00  0.00           H
HETATM    4 HH33 ACE A   1       1.602   2.139  -1.011  1.00  0.00           H
HETATM    5  C   ACE A   1       3.427   2.641   0.000  1.00  0.00           C
HETATM    6  O   ACE A   1       4.391   1.877   0.000  1.00  0.00           O
ATOM      7  N   ALA A   2       3.555   3.970   0.000  1.00  0.00           N
ATOM      8  H   ALA A   2       2.739   4.561   0.010  1.00  0.00           H
ATOM      9  CA  ALA A   2       4.853   4.614   0.000  1.00  0.00           C
ATOM     10  HA  ALA A   2       5.403   4.319   0.895  1.00  0.00           H
ATOM     11  CB  ALA A   2       5.661   4.221  -1.232  1.00  0.00           C
ATOM     12  HB1 ALA A   2       5.074   4.399  -2.138  1.00  0.00           H
ATOM     13  HB2 ALA A   2       5.959   3.175  -1.204  1.00  0.00           H
ATOM     14  HB3 ALA A   2       6.587   4.790  -1.298  1.00  0.00           H
ATOM     15  C   ALA A   2       4.713   6.129   0.000  1.00  0.00           C
ATOM     16  O   ALA A   2       3.601   6.653   0.000  1.00  0.00           O
HETATM   17  N   NME A   3       5.846   6.835   0.000  1.00  0.00           N
HETATM   18  H   NME A   3       6.729   6.342   0.018  1.00  0.00           H
HETATM   19  CH3 NME A   3       5.846   8.284   0.000  1.00  0.00           C
HETATM   20 HH31 NME A   3       4.852   8.684  -0.209  1.00  0.00           H
HETATM   21 HH32 NME A   3       6.533   8.656  -0.761  1.00  0.00           H
HETATM   22 HH33 NME A   3       6.164   8.650   0.977  1.00  0.00           H
TER
CONECT    2    1    3    4    5
CONECT    5    6    7
CONECT   15   17
CONECT   17   18   19
CONECT   19   20   21   22
END
"""


def write_fixture(path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ACE_ALA_NME_PDB)
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "RUNS/ala_dipeptide_validation/ace_ala_nme.pdb"
    p = write_fixture(target)
    print(f"wrote {p}")

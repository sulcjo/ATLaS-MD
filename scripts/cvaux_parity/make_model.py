# scripts/cvaux_parity/make_model.py  (run from the repo root: PYTHONPATH=. python scripts/cvaux_parity/make_model.py <RUN_P>/01_solvated_start.pdb <M.json>)
# Copied from the Task 14 CPU control (Stage C); uses the test fixture payload builder.
import sys
sys.path.insert(0, "tests")
from openmm import app
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import canonical_topology_sha256
from gareus.energy_decomposition import peptide_atom_groups_from_topology
from gareus.mbar_analysis.thermo_frames import backbone_torsion_quads

top = app.PDBFile(sys.argv[1]).topology
quads, labels = backbone_torsion_quads(top, peptide_atom_groups_from_topology(top, "all-peptide")[1])
blocks = [lab.split("-")[0] for lab in labels]
coeffs = [1.0] + [0.0] * (2 * len(quads) - 1)
draft = AuxModel.from_mapping(model_payload(quads, coeffs, blocks=blocks, label="stage-c-e2e"))
sha = canonical_topology_sha256(top, draft)
AuxModel.from_mapping(model_payload(quads, coeffs, blocks=blocks, label="stage-c-e2e",
                                    topology_sha256=sha)).write(sys.argv[2])
print(sha, labels)

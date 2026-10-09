from pathlib import Path
import numpy as np
import mdtraj as md

from gareus.adaptive.aux_discovery.descriptors import (descriptor_definition, evaluate_descriptors,
                                                       LAMBDA_NM, R0_HC_NM, R0_HB_NM)

PDB = Path(__file__).parent / "data" / "chignolin_solute.pdb"


def _traj():
    return md.load(str(PDB))


def test_c10_family_sizes_for_chignolin():
    d = descriptor_definition(_traj().topology)
    assert len(d.phi_labels) == 9 and len(d.psi_labels) == 9
    assert len(d.hc_labels) == 28 and len(d.hb_labels) == 63
    assert d.phi_labels[0] == "phi_TYR2" and d.psi_labels[0] == "psi_GLY1"


def test_atom_metadata_lengths_and_quad_names():
    t = _traj(); d = descriptor_definition(t.topology)
    n = t.topology.n_atoms
    assert len(d.atom_names) == n and len(d.atom_residue_names) == n and len(d.atom_residue_index) == n
    assert tuple(d.atom_names[i] for i in d.phi_quads[0]) == ("C", "N", "CA", "C")
    assert d.atom_residue_names[d.phi_quads[0][2]] == "TYR"


def test_torsions_match_mdtraj_iupac():
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    _, phi = md.compute_phi(t); _, psi = md.compute_psi(t)
    ang = np.concatenate([phi, psi], axis=1)
    np.testing.assert_allclose(out["tors_theta_iupac"], ang, atol=1e-6)
    np.testing.assert_allclose(out["tors"][:, 0::2], np.sin(ang), atol=1e-6)
    np.testing.assert_allclose(out["tors"][:, 1::2], np.cos(ang), atol=1e-6)


def test_contact_softmin_switch_formula():
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    a, b = d.hc_groups[0]
    x = np.linalg.norm(t.xyz[0, a][:, None, :] - t.xyz[0, b][None, :, :], axis=-1).ravel()
    m = x.min(); dist = m - LAMBDA_NM * np.log(np.exp(-(x - m) / LAMBDA_NM).sum())
    assert np.isclose(out["hc"][0, 0], 1.0 / (1.0 + (dist / R0_HC_NM) ** 6), atol=1e-6)


def test_hbond_switch_formula():
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    i, j = d.hb_pairs[0]
    r = np.linalg.norm(t.xyz[0, i] - t.xyz[0, j])
    assert np.isclose(out["hb"][0, 0], 1.0 / (1.0 + (r / R0_HB_NM) ** 6), atol=1e-6)


def test_basin_codes_match_discovery_census():
    from gareus.adaptive.discovery_census import basin_codes
    t = _traj(); d = descriptor_definition(t.topology)
    out = evaluate_descriptors(t.xyz, d)
    phi = np.degrees(out["tors_theta_iupac"][:, :9]); psi = np.degrees(out["tors_theta_iupac"][:, 9:])
    exp = basin_codes(phi[:, 0:8], psi[:, 1:9])
    np.testing.assert_array_equal(out["basin"], exp)


def test_schema_sha_is_stable():
    t = _traj()
    assert descriptor_definition(t.topology).schema_sha256 == descriptor_definition(t.topology).schema_sha256

"""Swarm residue contact map (contact-map CV1 spec step 1, ``gareus.swarm.contact_map``)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gareus.swarm import contact_map as CM
from gareus.swarm.members import run_member_loop

app = pytest.importorskip("openmm.app")


def _topology(n_res=10, heavy=(4, 2), hydrogens=2, extra_water=True):
    """A peptide-like topology: residue k has heavy[k % 2] heavy atoms and some hydrogens."""
    top = app.Topology()
    chain = top.addChain()
    C, H, O = app.Element.getBySymbol("C"), app.Element.getBySymbol("H"), app.Element.getBySymbol("O")
    for k in range(n_res):
        res = top.addResidue("ALA" if k % 2 else "GLY", chain, id=str(k + 1))
        for a in range(heavy[k % 2]):
            top.addAtom(f"C{a}", C, res)
        for h in range(hydrogens):
            top.addAtom(f"H{h}", H, res)
    if extra_water:
        w = top.addResidue("HOH", top.addChain())
        top.addAtom("O", O, w)
    return top


def test_definition_pairs_heavy_atoms_and_digest():
    top = _topology()
    d = CM.contact_map_definition(top)
    assert d["schema"] == CM.SCHEMA and d["min_sequence_separation"] == 3
    assert len(d["residues"]) == 10                                  # water excluded
    assert len(d["pairs"]) == sum(10 - (i + 3) for i in range(7))    # 28 for 10 residues, sep 3
    atoms = list(top.atoms())
    for p in d["pairs"]:
        assert p["j"] - p["i"] >= 3
        for a in p["atoms_i"] + p["atoms_j"]:
            assert atoms[a].element.symbol != "H"
    assert CM.contact_map_definition(top)["sha256"] == d["sha256"]
    assert CM.contact_map_definition(top, min_sequence_separation=2)["sha256"] != d["sha256"]
    with pytest.raises(ValueError):
        CM.contact_map_definition(top, lambda_angstrom=0.0)


def test_evaluator_matches_the_reference_and_respects_the_soft_min_bounds():
    top = _topology()
    d = CM.contact_map_definition(top)
    ev = CM.ContactMapEvaluator(d)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.6, size=(top.getNumAtoms(), 3))             # nm
    got = ev(x)
    assert got.shape == (len(d["pairs"]),)
    for k, p in enumerate(d["pairs"]):
        r = np.linalg.norm(x[p["atoms_j"]][None, :, :] - x[p["atoms_i"]][:, None, :], axis=-1).ravel() * 10
        ref = CM.soft_min_distance(r, d["lambda_angstrom"])
        assert got[k] == pytest.approx(ref, rel=1e-12, abs=1e-12)
        lam, n = d["lambda_angstrom"], r.size
        assert r.min() - lam * np.log(n) - 1e-12 <= got[k] <= r.min() + 1e-12


def test_stable_for_far_apart_residues_where_the_naive_sum_underflows():
    top = _topology()
    d = CM.contact_map_definition(top, lambda_angstrom=0.02)
    x = np.random.default_rng(1).normal(scale=0.3, size=(top.getNumAtoms(), 3))
    x[: 6] += 50.0                                                      # first residues 500 A away
    got = CM.ContactMapEvaluator(d)(x)
    assert np.all(np.isfinite(got)) and got.max() > 400.0


def test_small_lambda_approaches_the_minimum_distance():
    top = _topology()
    x = np.random.default_rng(2).normal(scale=0.6, size=(top.getNumAtoms(), 3))
    d = CM.contact_map_definition(top, lambda_angstrom=1e-4)
    got = CM.ContactMapEvaluator(d)(x)
    for k, p in enumerate(d["pairs"]):
        r = np.linalg.norm(x[p["atoms_j"]][None] - x[p["atoms_i"]][:, None], axis=-1) * 10
        assert got[k] == pytest.approx(r.min(), abs=1e-3)


def test_index_roundtrip_and_tamper_detection(tmp_path):
    d = CM.contact_map_definition(_topology())
    path = tmp_path / CM.INDEX_NAME
    CM.write_index(path, d)
    assert CM.read_index(path) == d
    data = json.loads(path.read_text())
    data["lambda_angstrom"] = 0.5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        CM.read_index(path)


def test_member_loop_writes_the_extra_stream_row_aligned(tmp_path):
    state = {"t": 0}

    def step_fn(n):
        state["t"] += n

    def measure_fn():
        return {"cv1": state["t"] * 1e-3, "rg_nm": 0.5, "e2e_nm": 1.0, "v_pep_kj": 0.0, "v_dih_kj": 0.0,
                "potential_kj": 0.0}

    out = run_member_loop(
        n_equil_steps=10, n_prod_steps=40, steps_per_frame=10, seed_frame_every=0, step_fn=step_fn,
        measure_fn=measure_fn, write_frame_fn=lambda i: None, trace_path=tmp_path / "trace.csv",
        feature_fn=lambda: [state["t"], 0.0], features_path=tmp_path / "torsion_features.npy",
        extra_features=[(lambda: [state["t"] * 2.0, 1.0, 2.0], tmp_path / CM.FEATURES_NAME)])
    assert out["n_frames"] == 4
    tors = np.load(tmp_path / "torsion_features.npy")
    cmap = np.load(tmp_path / CM.FEATURES_NAME)
    assert tors.shape == (4, 2) and cmap.shape == (4, 3)
    assert np.allclose(cmap[:, 0], 2 * tors[:, 0])                      # same frame per row
    assert not (tmp_path / "contact_map_features.tmp.npy").exists()


def test_chignolin_topology_gives_28_pairs_when_available():
    from pathlib import Path
    pdb = Path.home() / ".claude/jobs/6bfa4c5c/tmp/modality/mixture/swarm/system/topology.pdb"
    if not pdb.exists():
        pytest.skip("chignolin swarm topology not available")
    d = CM.contact_map_definition(app.PDBFile(str(pdb)).topology)
    assert len(d["residues"]) == 10 and len(d["pairs"]) == 28


def test_a_peptide_too_short_for_any_pair_gives_an_empty_map_not_an_error(tmp_path):
    top = _topology(n_res=2)
    d = CM.contact_map_definition(top)
    assert d["pairs"] == []
    row = CM.ContactMapEvaluator(d)(np.zeros((top.getNumAtoms(), 3)))
    assert row.shape == (0,)
    out = run_member_loop(
        n_equil_steps=0, n_prod_steps=20, steps_per_frame=10, seed_frame_every=0, step_fn=lambda n: None,
        measure_fn=lambda: {"cv1": 0.0}, write_frame_fn=lambda i: None, trace_path=tmp_path / "trace.csv",
        extra_features=[(lambda: CM.ContactMapEvaluator(d)(np.zeros((top.getNumAtoms(), 3))),
                         tmp_path / CM.FEATURES_NAME)])
    assert out["n_frames"] == 2 and np.load(tmp_path / CM.FEATURES_NAME).shape == (2, 0)


def test_reference_soft_min_is_stable_for_far_pairs():
    assert CM.soft_min_distance([500.0, 501.0], 0.02) == pytest.approx(500.0, abs=1e-6)


def test_evaluator_refuses_an_edited_definition():
    d = CM.contact_map_definition(_topology())
    with pytest.raises(ValueError, match="digest"):
        CM.ContactMapEvaluator({**d, "lambda_angstrom": 0.5})


def test_pdb_style_hydrogen_names_without_element_are_hydrogens():
    class _A:
        def __init__(self, name):
            self.name, self.element = name, None
    assert all(CM._is_hydrogen(_A(n)) for n in ("H", "HA", "1HB", "2HD1"))
    assert not any(CM._is_hydrogen(_A(n)) for n in ("CA", "N", "OG1", "CH2"))


def test_member_loader_checks_shape_and_tolerates_missing_files(tmp_path):
    d = CM.contact_map_definition(_topology())
    assert CM.load_member_contact_map(tmp_path) is None
    CM.write_index(tmp_path / CM.INDEX_NAME, d)
    np.save(tmp_path / CM.FEATURES_NAME, np.zeros((5, len(d["pairs"]))))
    X, dd = CM.load_member_contact_map(tmp_path)
    assert X.shape == (5, 28) and dd == d
    np.save(tmp_path / CM.FEATURES_NAME, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="does not match"):
        CM.load_member_contact_map(tmp_path)


def test_cli_rejects_bad_contact_map_settings_and_keeps_zero_out(tmp_path):
    from gareus.cli import parse_args
    base = ["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")]
    a = parse_args(base)
    assert (a.swarm_contact_map_min_separation, a.swarm_contact_map_lambda_a) == (3, 0.2)
    for bad in (["--swarm-contact-map-min-separation", "0"], ["--swarm-contact-map-lambda-a", "-0.2"],
                ["--swarm-contact-map-lambda-a", "0"]):
        with pytest.raises(SystemExit):
            parse_args(base + bad)

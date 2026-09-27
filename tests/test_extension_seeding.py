"""Frozen-final extensions continue from parent end states instead of re-pulling.

chignolin_9's final_extension_001 (job 2680578) spent 51 % of its wall re-grafting
and US-pulling every window of the unchanged final window set.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from gareus import extension_seeding as es  # noqa: E402
from gareus.production import _seed_extension_windows_from_parent_ends  # noqa: E402
from gareus.topup_seeding import DIR_NAME, SeedMismatchError, SeedState, should_export_final_window_states  # noqa: E402

N_ATOMS = 2
PDB_TEMPLATE = (
    "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
    "ATOM      1  C1  UNK A   1    {x:8.3f}   1.000   1.000  1.00  0.00           C\n"
    "ATOM      2  C2  UNK A   1       2.000   2.000   2.000  1.00  0.00           C\n"
    "END\n"
)


def _window_map(d: Path, state_ids):
    rows = ["epoch_window,state_id"] + [f"{w},{sid}" for w, sid in enumerate(state_ids)]
    (d / "epoch_window_map.csv").write_text("\n".join(rows) + "\n")


def _manifest(d: Path, centers, ks, sec_c=None, sec_k=None):
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    (d / "checkpoints" / "production_checkpoint_manifest.json").write_text(json.dumps({
        "windows_A": centers, "window_k_kcal_mol_A2": ks,
        "secondary_cv_centers": sec_c, "secondary_cv_k_kcal_mol": sec_k,
    }))


def _pdb_parent(d: Path, state_ids, centers, ks, x=1.0):
    """A top-ups-off segment: final_pdbs + window map + checkpoint manifest, no State export."""
    (d / "final_pdbs").mkdir(parents=True, exist_ok=True)
    _window_map(d, state_ids)
    _manifest(d, centers, ks)
    for w in range(len(state_ids)):
        # replica index deliberately != window index, as after exchanges
        (d / "final_pdbs" / f"replica_{(w + 1) % len(state_ids):03d}_window_{w:03d}.pdb").write_text(
            PDB_TEMPLATE.format(x=x + w))


def _state_xml(x: float) -> str:
    system = openmm.System()
    for _ in range(N_ATOMS):
        system.addParticle(12.0)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([openmm.Vec3(x, 0, 0), openmm.Vec3(0, 0, 0)] * unit.nanometer)
    st = ctx.getState(getPositions=True, getVelocities=True)
    return openmm.XmlSerializer.serialize(st)


def _export_parent(d: Path, records):
    """records: sid -> (x, primary_center_nm, primary_k_kj)."""
    (d / DIR_NAME).mkdir(parents=True, exist_ok=True)
    index = {}
    for sid, (x, c, k) in records.items():
        (d / DIR_NAME / f"state_{sid}.xml").write_text(_state_xml(x))
        index[str(sid)] = {"window": 0, "cv1": 0.0, "cv2": float("nan"), "primary_center": c, "primary_k": k,
                           "secondary_center": None, "secondary_k": None, "export_seq": 1}
    (d / DIR_NAME / "index.json").write_text(json.dumps(index))


def test_parent_dirs_are_final_segments_then_earlier_extensions(tmp_path):
    ap = tmp_path
    base = ap / "final" / "baseline"
    t_late, t_early = ap / "final" / "topup_001_100", ap / "final" / "topup_002_900"
    for d in (base, t_late, t_early):
        (d / "final_pdbs").mkdir(parents=True)
    os.utime(t_early / "final_pdbs", ns=(1_000, 1_000))
    os.utime(t_late / "final_pdbs", ns=(2_000, 2_000))
    for k in (1, 2, 3):
        (ap / f"final_extension_{k:03d}").mkdir()
    # round 3 (ext_index 2) sees extensions 1 and 2, never itself
    assert es.extension_parent_dirs(ap, 2) == [
        base, t_early, t_late, ap / "final_extension_001", ap / "final_extension_002"]
    assert es.extension_parent_dirs(ap, 0) == [base, t_early, t_late]


def test_topup_order_prefers_export_seq_over_mtime(tmp_path):
    base = tmp_path / "final" / "baseline"
    a, b = tmp_path / "final" / "topup_001_5", tmp_path / "final" / "topup_002_9"
    for d, seq in ((a, 200), (b, 100)):
        (d / DIR_NAME).mkdir(parents=True)
        (d / DIR_NAME / "index.json").write_text(json.dumps({"0": {"export_seq": seq}}))
    base.mkdir(parents=True)
    os.utime(a / DIR_NAME / "index.json", ns=(1, 1))      # a copy scrambled a's mtime
    assert es.extension_parent_dirs(tmp_path, 0) == [base, b, a]


def test_flat_final_phase_is_its_own_parent(tmp_path):
    (tmp_path / "final" / "final_pdbs").mkdir(parents=True)
    assert es.extension_parent_dirs(tmp_path, 0) == [tmp_path / "final"]


def test_pdb_seeds_resolve_state_through_parent_window_map(tmp_path):
    parent = tmp_path / "p"
    _pdb_parent(parent, state_ids=[7, 3], centers=[0.1, 0.2], ks=[10.0, 20.0])
    seeds = es.pdb_seeds_of_parent(parent)
    assert set(seeds) == {7, 3}
    assert seeds[3].path.name.endswith("_window_001.pdb")
    assert (seeds[3].primary_center, seeds[3].primary_k) == (0.2, 20.0)


def test_newer_pdb_only_parent_beats_older_export(tmp_path):
    old, new = tmp_path / "final" / "baseline", tmp_path / "final_extension_001"
    _export_parent(old, {0: (1.0, 0.1, 5.0), 1: (1.0, 0.2, 5.0)})
    _pdb_parent(new, state_ids=[0], centers=[0.1], ks=[10.0])
    sources = es.resolve_seed_sources([old, new], [0, 1])
    assert sources[0] == ("pdb", new)
    assert sources[1] == ("export", old)          # only the older parent holds state 1


def test_export_beats_pdb_within_one_parent(tmp_path):
    d = tmp_path / "seg"
    _pdb_parent(d, state_ids=[0], centers=[0.1], ks=[10.0])
    _export_parent(d, {0: (1.0, 0.1, 5.0)})
    assert es.resolve_seed_sources([d], [0]) == {0: ("export", d)}
    seeds = es.load_extension_seeds([d], [0])
    assert isinstance(seeds[0], SeedState)


def test_extension_exports_states_even_with_topups_off():
    ext = SimpleNamespace(adaptive_production_topups=False, _adaptive_phase_info={"is_extension": True})
    plain = SimpleNamespace(adaptive_production_topups=False, _adaptive_phase_info={"is_adaptive_epoch": True})
    assert should_export_final_window_states(ext) is True
    assert should_export_final_window_states(plain) is False


def _ext_args(parents):
    return SimpleNamespace(_adaptive_phase_info={"is_extension": True, "extension_parent_dirs": [str(p) for p in parents]})


def _call(ext_dir, parents, centers_a, k_list, n_atoms=N_ATOMS):
    n = len(centers_a)
    return _seed_extension_windows_from_parent_ends(
        _ext_args(parents), ext_dir, n, n_atoms,
        [c / 10 for c in centers_a], [k * 418.4 for k in k_list], None, None,
        centers_a, k_list, None,
    )


def test_production_helper_seeds_every_window_from_pdbs(tmp_path):
    parent, ext = tmp_path / "final_extension_001", tmp_path / "final_extension_002"
    _pdb_parent(parent, state_ids=[4, 9], centers=[0.1, 0.2], ks=[10.0, 20.0], x=3.0)
    ext.mkdir()
    _window_map(ext, [9, 4])                        # extension's own window order differs
    pos, vel, boxes, state_seeds, pdb_windows = _call(ext, [parent], [0.2, 0.1], [20.0, 10.0])
    assert pdb_windows == {0, 1} and state_seeds == {}
    assert vel == [None, None]
    # ext window 0 is state 9 = parent window 1, whose PDB has x = 3.0 + 1 A
    assert pos[0][0].value_in_unit(unit.angstrom)[0] == pytest.approx(4.0)
    assert boxes[0][0].value_in_unit(unit.angstrom)[0] == pytest.approx(30.0)


def test_production_helper_returns_none_when_a_window_has_no_parent_state(tmp_path):
    parent, ext = tmp_path / "p", tmp_path / "e"
    _pdb_parent(parent, state_ids=[4], centers=[0.1], ks=[10.0])
    ext.mkdir()
    _window_map(ext, [4, 5])
    assert _call(ext, [parent], [0.1, 0.3], [10.0, 10.0]) is None


def test_production_helper_returns_none_without_own_window_map(tmp_path):
    parent, ext = tmp_path / "p", tmp_path / "e"
    _pdb_parent(parent, state_ids=[0], centers=[0.1], ks=[10.0])
    ext.mkdir()
    assert _call(ext, [parent], [0.1], [10.0]) is None


def test_production_helper_rejects_restraint_mismatch(tmp_path):
    parent, ext = tmp_path / "p", tmp_path / "e"
    _pdb_parent(parent, state_ids=[0], centers=[0.1], ks=[10.0])
    ext.mkdir()
    _window_map(ext, [0])
    with pytest.raises(SeedMismatchError, match="primary_center"):
        _call(ext, [parent], [0.15], [10.0])


def test_production_helper_rejects_wrong_atom_count(tmp_path):
    parent, ext = tmp_path / "p", tmp_path / "e"
    _pdb_parent(parent, state_ids=[0], centers=[0.1], ks=[10.0])
    ext.mkdir()
    _window_map(ext, [0])
    with pytest.raises(SeedMismatchError, match="atoms"):
        _call(ext, [parent], [0.1], [10.0], n_atoms=19008)

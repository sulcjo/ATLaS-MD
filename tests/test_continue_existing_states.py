"""--ap-continue-states: numbered epochs >= 1 and the final phase continue every existing state from its
newest earlier end state; only states with no earlier end state (new ones) are seeded and pulled.

chignolin_9 analysis (2026-10-01): every numbered epoch re-grafted and re-pulled all 236 windows from
seeds, so continuous trajectories lasted ~14 ns while the D3/P4 turn dihedrals decorrelate on >> 10 ns
(ACF 0.93 at 10 ns) -- the fold populations stayed seed-locked.
"""
from pathlib import Path

import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from gareus import extension_seeding as es  # noqa: E402
from gareus.production import _continue_windows_from_parent_ends, merge_partial_pull  # noqa: E402
from gareus.topup_seeding import SeedMismatchError  # noqa: E402

from test_extension_seeding import _pdb_parent, _window_map, N_ATOMS  # noqa: E402


def _mk(p: Path, with_pdbs=True):
    p.mkdir(parents=True, exist_ok=True)
    if with_pdbs:
        (p / "final_pdbs").mkdir(exist_ok=True)
    return p


# --------------------------------------------------------------------------- parent ordering

def test_prior_phase_parents_are_all_earlier_phase_segments_oldest_first(tmp_path):
    ad = tmp_path
    e0 = _mk(ad / "epoch_000")
    e1b = _mk(ad / "epoch_001" / "baseline")
    e1t = _mk(ad / "epoch_001" / "topup_001_5000")
    cur = _mk(ad / "epoch_002" / "baseline", with_pdbs=False)
    _mk(ad / "epoch_003" / "baseline")                       # later epoch: never a parent
    assert es.prior_phase_parent_dirs(ad, cur) == [e0, e1b, e1t]
    fin = _mk(ad / "final" / "baseline", with_pdbs=False)
    assert es.prior_phase_parent_dirs(ad, fin)[-1] == ad / "epoch_003" / "baseline"
    assert es.prior_phase_parent_dirs(ad, ad / "epoch_000") == []


def test_prior_phase_parents_skip_phases_without_end_states(tmp_path):
    ad = tmp_path
    _mk(ad / "epoch_000", with_pdbs=False)                    # never finished: no end states
    e1 = _mk(ad / "epoch_001" / "baseline")
    assert es.prior_phase_parent_dirs(ad, ad / "epoch_002" / "baseline") == [e1]


# --------------------------------------------------------------------------- driver helper

def test_driver_passes_parents_only_when_enabled_for_baseline_of_later_phases(tmp_path):
    from gareus.adaptive_production import continuation_parent_dirs
    ad = tmp_path
    e0 = _mk(ad / "epoch_000")
    ep1 = ad / "epoch_001"
    assert continuation_parent_dirs(ad, ep1, "baseline", enabled=True) == [str(e0)]
    assert continuation_parent_dirs(ad, ep1, "baseline", enabled=False) == []
    assert continuation_parent_dirs(ad, ep1, "topup_001_5000", enabled=True) == []   # top-ups seed themselves
    assert continuation_parent_dirs(ad, ad / "epoch_000", "baseline", enabled=True) == []
    assert continuation_parent_dirs(ad, ad / "final", "baseline", enabled=True) == [str(e0)]


# --------------------------------------------------------------------------- production helper

def _call(phase_dir, parents, centers_a, k_list, n_atoms=N_ATOMS):
    from types import SimpleNamespace
    args = SimpleNamespace(_adaptive_phase_info={"continue_parent_dirs": [str(p) for p in parents]})
    return _continue_windows_from_parent_ends(
        args, phase_dir, len(centers_a), n_atoms,
        [c / 10 for c in centers_a], [k * 418.4 for k in k_list], None, None,
        centers_a, k_list, None,
    )


def test_existing_states_continue_and_new_states_are_reported_missing(tmp_path):
    parent, cur = tmp_path / "epoch_001" / "baseline", tmp_path / "epoch_002" / "baseline"
    _pdb_parent(parent, state_ids=[4, 9], centers=[0.1, 0.2], ks=[10.0, 20.0], x=3.0)
    cur.mkdir(parents=True)
    _window_map(cur, [9, 4, 17])                               # state 17 was added by this epoch's actions
    pos, vel, boxes, state_seeds, pdb_windows, missing = _call(cur, [parent], [0.2, 0.1, 0.3], [20.0, 10.0, 15.0])
    assert missing == [2] and pdb_windows == {0, 1}
    assert pos[2] is None and boxes[2] is None
    assert pos[0][0].value_in_unit(unit.angstrom)[0] == pytest.approx(4.0)     # state 9 = parent window 1


def test_no_parents_means_every_window_is_missing(tmp_path):
    cur = tmp_path / "epoch_001" / "baseline"; cur.mkdir(parents=True)
    _window_map(cur, [0, 1])
    *_, pdb_windows, missing = _call(cur, [], [0.1, 0.2], [10.0, 10.0])
    assert missing == [0, 1] and pdb_windows == set()


def test_a_parent_with_a_different_restraint_for_the_same_state_is_refused(tmp_path):
    parent, cur = tmp_path / "a", tmp_path / "b"
    _pdb_parent(parent, state_ids=[0], centers=[0.1], ks=[10.0])
    cur.mkdir(); _window_map(cur, [0])
    with pytest.raises(SeedMismatchError):
        _call(cur, [parent], [0.15], [10.0])


# --------------------------------------------------------------------------- merge of a subset pull

def test_merge_partial_pull_places_pulled_windows_and_maps_drops_back():
    continued_pos = ["c0", None, "c2", None]
    continued_vel = ["v0", None, None, None]
    pos, vel, dropped = merge_partial_pull(continued_pos, continued_vel, missing=[1, 3],
                                           pulled_pos=["p1", "p3"], pulled_vel=["pv1", "pv3"], pulled_dropped=[1])
    assert pos == ["c0", "p1", "c2", "p3"] and vel == ["v0", "pv1", None, "pv3"]
    assert dropped == [3]
    with pytest.raises(ValueError):
        merge_partial_pull(continued_pos, continued_vel, missing=[1, 3], pulled_pos=["p1"], pulled_vel=["x"], pulled_dropped=[])


# --------------------------------------------------------------------------- CLI

def test_cli_flag_defaults_off():
    from gareus.cli import parse_args
    assert parse_args(["--seq", "GYDPETGTWG"]).ap_continue_states is False
    assert parse_args(["--seq", "GYDPETGTWG", "--ap-continue-states"]).ap_continue_states is True

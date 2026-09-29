"""X3 slow-mode-aware reseeding: hidden mode, admissibility, balance selection, overrides, off = no-op."""
import csv
import json
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import gareus.adaptive_production as ap
from gareus.adaptive import slow_mode_reseed as X3
from gareus.adaptive import slow_mode_reseed_io as IO
from gareus.cv_selection.models import fit_residual_components
from gareus.cv_selection.slowness import fit_conditional_tica

RT = X3.R_KCAL * 300.0


# ------------------------------------------------------------------ synthetic hidden slow mode

def _synthetic(seed=0, n_members=4, n_frames=5000, d=8):
    """e0: a SLOW, unimodal, high-variance direction (the deployed CV2, like a tICA pick);
    e1: a two-state telegraph process switching rarely (the hidden slow mode);
    everything else fast noise. CV1 is independent of the torsions."""
    rng = np.random.default_rng(seed)
    X, anchor, member, frame, state = [], [], [], [], []
    for m in range(n_members):
        z0, s = 0.0, 1.0 if m % 2 else -1.0
        for t in range(n_frames):
            z0 = 0.999 * z0 + math.sqrt(1 - 0.999 ** 2) * 2.0 * rng.standard_normal()
            if rng.random() < 0.002:
                s = -s
            x = 0.3 * rng.standard_normal(d)
            x[0] += z0
            x[1] += s + 0.2 * rng.standard_normal()
            X.append(x); anchor.append(rng.random()); member.append(m); frame.append(t); state.append(s)
    return (np.asarray(X), np.asarray(anchor), np.asarray(member), np.asarray(frame), np.asarray(state))


@pytest.fixture(scope="module")
def synth():
    X, a, member, frame, s = _synthetic()
    fit = fit_residual_components(X, a, n_components=3)
    assert abs(fit.right_vectors[0][0]) > 0.99            # PC1 is the slow unimodal e0
    return fit, X, a, member, frame, s


def test_plain_conditional_tica_finds_the_deployed_direction(synth):
    fit, X, a, member, frame, _s = synth
    plain = fit_conditional_tica(X3._stripped(fit), X, a, member, frame, lag=10, n_modes=1)
    assert abs(plain.right_vectors[0][0]) > 0.9            # without removing CV2 it is CV2 again


def test_hidden_mode_is_the_two_state_mode_orthogonal_to_cv2(synth):
    fit, X, a, member, frame, s = synth
    mode = X3.fit_hidden_mode(fit, 1, X, a, member, frame, lag=10)
    assert abs(mode.direction[1]) > 0.95
    assert abs(mode.direction @ fit.right_vectors[0]) < 1e-8
    assert mode.eigenvalue > 0.8 and mode.split_method == "density_minimum"
    side = mode.side(mode.scores(X, a))
    agree = np.mean(side == (s > 0))
    assert max(agree, 1 - agree) > 0.97


def test_hidden_mode_is_deterministic(synth):
    fit, X, a, member, frame, _s = synth
    m1 = X3.fit_hidden_mode(fit, 1, X, a, member, frame, lag=10)
    m2 = X3.fit_hidden_mode(fit, 1, X, a, member, frame, lag=10)
    assert np.array_equal(m1.direction, m2.direction) and m1.split == m2.split


def test_split_point_density_minimum_and_median_fallback():
    rng = np.random.default_rng(1)
    bi = np.concatenate([rng.normal(-2, 0.4, 5000), rng.normal(2, 0.4, 3000)])
    b, method = X3.split_point(bi)
    assert method == "density_minimum" and -1.0 < b < 1.0
    uni = rng.normal(0.3, 1.0, 8000)
    b, method = X3.split_point(uni)
    assert method == "median" and b == pytest.approx(float(np.median(uni)))


def test_side_occupancy_counts_and_minority():
    occ = X3.side_occupancy([1, 1, 1, 0, 0, 0, 0, 0], [5, 5, 5, 5, 7, 7, 7, 7])
    assert occ[5]["n_high"] == 3 and occ[5]["minority_side"] == 0 and occ[5]["minority_fraction"] == 0.25
    assert occ[7]["frac_high"] == 0.0 and occ[7]["minority_side"] == 1


# ------------------------------------------------------------------ admissibility

def test_admissibility_uses_sigma_w_per_restrained_axis():
    sig1 = math.sqrt(RT / 400.0)
    t = {"primary_center": 0.3, "primary_k": 400.0, "secondary_center": 1.0, "secondary_k": 0.0}
    assert X3.admissible(0.3 + 1.9 * sig1, 99.0, t, RT)          # CV2 unrestrained: ignored
    assert not X3.admissible(0.3 + 2.1 * sig1, 1.0, t, RT)
    t2 = dict(t, secondary_k=2.0)
    sig2 = math.sqrt(RT / 2.0)
    assert X3.admissible(0.3, 1.0 + 1.5 * sig2, t2, RT)
    assert not X3.admissible(0.3, 1.0 + 2.5 * sig2, t2, RT)
    assert not X3.admissible(0.3, None, t2, RT)                  # restrained CV2 unmeasured
    assert X3.restraint_distance(5.0, None, {"primary_center": 0.0, "primary_k": 0.0}, RT) == 0.0


# ------------------------------------------------------------------ selection

def _targets(n=8):
    return [{"state_id": i, "primary_center": 0.1 * i, "primary_k": 400.0, "secondary_center": None,
             "secondary_k": None, "gamd_lambda": 0.0} for i in range(n)]


def _occ(minority):
    return {i: {"n_frames": 500, "frac_high": f, "minority_side": 1 if f < 0.5 else 0,
                "minority_fraction": min(f, 1 - f)} for i, f in minority.items()}


def _cand(path, cv1, side, src=0):
    return {"path": path, "cv1": cv1, "cv2": None, "side": side, "hidden_z": 1.0 if side else -1.0,
            "source_state_id": src, "source_phase": "p"}


def test_plan_picks_most_one_sided_windows_and_minority_side_seeds():
    targets = _targets(8)
    occ = _occ({0: 0.02, 1: 0.10, 2: 0.95, 3: 0.40, 4: 0.5, 5: 0.15, 6: 0.3, 7: 0.99})
    sig = math.sqrt(RT / 400.0)
    cands = [_cand(f"c{i}_{s}", 0.1 * i + 0.5 * sig, s, src=i) for i in range(8) for s in (0, 1)]
    plan = X3.plan_reseed(targets, occ, cands, fraction=0.25, rt_kcal=RT)
    assert plan["n_target"] == 2 and plan["n_one_sided"] == 5
    chosen = [r["target_state_id"] for r in plan["reseeded"]]
    assert chosen == [7, 0]                                      # minority 0.01, 0.02
    for r in plan["reseeded"]:
        assert r["seed_side"] == r["minority_side"] and r["restraint_distance_sigma"] <= 2.0
    assert plan["reseeded"][0]["seed_side"] == 0 and plan["reseeded"][1]["seed_side"] == 1


def test_window_without_admissible_candidate_is_skipped_and_the_next_takes_its_slot():
    targets = _targets(4)
    occ = _occ({0: 0.01, 1: 0.05, 2: 0.10, 3: 0.5})
    sig = math.sqrt(RT / 400.0)
    cands = [_cand("far", 0.0 + 5 * sig, 1), _cand("ok1", 0.1, 1), _cand("ok2", 0.2, 1)]
    plan = X3.plan_reseed(targets, occ, cands, fraction=0.5, rt_kcal=RT)
    assert [r["target_state_id"] for r in plan["reseeded"]] == [1, 2]
    assert plan["skipped"] == [{"state_id": 0, "reason": "no_admissible_minority_side_candidate",
                                "minority_side": 1, "minority_fraction": 0.01}]


def test_default_seed_already_on_minority_side_is_not_reseeded():
    targets = _targets(3)
    occ = _occ({0: 0.01, 1: 0.05, 2: 0.5})
    cands = [_cand("a", 0.0, 1), _cand("b", 0.1, 1)]
    plan = X3.plan_reseed(targets, occ, cands, fraction=0.34, rt_kcal=RT, default_sides={0: 1})
    assert [r["target_state_id"] for r in plan["reseeded"]] == [1]
    assert plan["skipped"][0]["reason"] == "default_seed_already_on_minority_side"


def test_plan_is_deterministic_under_candidate_order_and_prefers_unused_seeds():
    targets = [dict(t, primary_center=0.1) for t in _targets(4)]
    occ = _occ({0: 0.01, 1: 0.02, 2: 0.03, 3: 0.04})
    cands = [_cand("near", 0.1, 1), _cand("farther", 0.1 + 0.01, 1)]
    plans = []
    for seed in range(3):
        shuffled = list(cands)
        random.Random(seed).shuffle(shuffled)
        plans.append(X3.plan_reseed(targets, occ, shuffled, fraction=1.0, rt_kcal=RT))
    assert all(p["reseeded"] == plans[0]["reseeded"] for p in plans)
    assert [r["seed_path"] for r in plans[0]["reseeded"]] == ["near", "farther", "near", "farther"]


def test_too_few_frames_is_reported():
    occ = {0: {"n_frames": 10, "frac_high": 0.0, "minority_side": 1, "minority_fraction": 0.0}}
    plan = X3.plan_reseed(_targets(1), occ, [], fraction=1.0, rt_kcal=RT)
    assert plan["skipped"][0]["reason"] == "too_few_frames" and plan["n_reseeded"] == 0


# ------------------------------------------------------------------ overrides and the seeding lookup

def _bank_with_overrides(tmp_path):
    bank = tmp_path / "adaptive_production" / "seed_bank_epoch_000"
    (bank / "pdbs").mkdir(parents=True)
    (bank / "final_survivor_seeds.csv").write_text("seed_name,survivor_pdb_path\n")
    src = tmp_path / "end.pdb"
    src.write_text("END\n")
    reseeded = [{"target_state_id": s, "target_primary_center": 0.1 * s, "target_primary_k": 400.0,
                 "target_secondary_center": None, "target_secondary_k": None, "seed_path": str(src),
                 "seed_source_state_id": 9, "seed_cv1": 0.1 * s, "seed_cv2": None, "seed_hidden_z": 1.2,
                 "seed_side": 1} for s in (1, 3)]
    X3.write_overrides(bank, reseeded)
    return bank


def _fake_loader(d):
    rows = list(csv.DictReader((d / "final_survivor_seeds.csv").open()))
    return [{"pdb_path": d / r["survivor_pdb_path"], "source_row": r} for r in rows]


def test_overrides_round_trip_through_filter_and_window_lookup(tmp_path):
    bank = _bank_with_overrides(tmp_path)
    assert sorted(p.name for p in (bank / X3.DIR_NAME / "pdbs").iterdir()) == \
        ["x3_state_0001_000.pdb", "x3_state_0003_001.pdb"]
    filtered = tmp_path / "seg" / "filtered_seed_bank"
    filtered.mkdir(parents=True)
    assert X3.filter_overrides(bank, [0, 1, 2], filtered) == 1
    phase = tmp_path / "seg"
    (phase / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n1,1\n2,2\n")
    got = X3.reseed_conformers_by_window(filtered, phase, [0.0, 0.1, 0.2], None, _fake_loader)
    assert list(got) == [1] and got[1]["source_row"]["target_state_id"] == "1"
    # a window whose restraint centre disagrees keeps its default seed
    assert X3.reseed_conformers_by_window(filtered, phase, [0.0, 0.15, 0.2], None, _fake_loader) == {}


def test_no_override_dir_means_no_lookup(tmp_path):
    called = []
    assert X3.reseed_conformers_by_window(tmp_path, tmp_path, [0.0], None, lambda d: called.append(d)) == {}
    assert called == []


def test_filter_seed_bank_is_unchanged_without_overrides_and_carries_them_with(tmp_path):
    bank = tmp_path / "adaptive_production" / "seed_bank_epoch_000"
    (bank / "pdbs").mkdir(parents=True)
    pdb = bank / "pdbs" / "seed_0.pdb"
    pdb.write_text("END\n")
    with (bank / "final_survivor_seeds.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["seed_name", "survivor_pdb_path", "source_state_id"])
        w.writeheader()
        w.writerow({"seed_name": "seed_0", "survivor_pdb_path": str(pdb), "source_state_id": "1"})
    off = ap.filter_seed_bank_for_state_ids(bank, [1], tmp_path / "off")
    assert "slow_mode_reseed_rows" not in off and not (tmp_path / "off" / X3.DIR_NAME).exists()
    src = tmp_path / "end.pdb"
    src.write_text("END\n")
    X3.write_overrides(bank, [{"target_state_id": 1, "target_primary_center": 0.1, "seed_path": str(src),
                               "seed_cv1": 0.1, "seed_hidden_z": 1.0, "seed_side": 1}])
    on = ap.filter_seed_bank_for_state_ids(bank, [1], tmp_path / "on")
    assert on["slow_mode_reseed_rows"] == 1 and (tmp_path / "on" / X3.DIR_NAME / "final_survivor_seeds.csv").exists()
    assert [r["seed_name"] for r in on["rows"]] == [r["seed_name"] for r in off["rows"]]


# ------------------------------------------------------------------ flag, policy, runner

def test_flag_defaults_off_reaches_the_policy_and_is_frozen():
    from gareus.cli import parse_args

    assert ap.policy_from_args(parse_args(["--seq", "DPETG"])).slow_mode_reseed_fraction == 0.0
    p = ap.policy_from_args(parse_args(["--seq", "DPETG", "--ap-slow-mode-reseed-fraction", "0.25"]))
    assert p.slow_mode_reseed_fraction == 0.25
    assert "slow_mode_reseed_fraction" in ap.DECISION_SETTINGS_FIELDS
    with pytest.raises(SystemExit):
        parse_args(["--seq", "DPETG", "--ap-slow-mode-reseed-fraction", "1.5"])


def test_runner_skips_without_a_frozen_pair_and_never_raises(tmp_path):
    bank = _bank_with_overrides(tmp_path)
    epoch = tmp_path / "adaptive_production" / "epoch_000"
    epoch.mkdir()
    args = SimpleNamespace(secondary_cv_model=None, secondary_cv_candidate_set=None,
                           secondary_cv_feature_schema=None, temperature_k=300.0)
    rep = IO.run_epoch_slow_mode_reseed(fraction=0.25, epoch_dir=epoch, phase_dirs=[epoch], states=[],
                                        seed_bank_dir=bank, args=args)
    assert rep["status"] == "skipped" and not (bank / X3.DIR_NAME).exists()   # stale override removed
    assert json.loads((epoch / X3.REPORT_NAME).read_text())["status"] == "skipped"


def test_runner_error_clears_overrides_and_returns(tmp_path, monkeypatch):
    bank = _bank_with_overrides(tmp_path)
    epoch = tmp_path / "adaptive_production" / "epoch_000"
    epoch.mkdir()
    (epoch / "solute_only.pdb").write_text("END\n")
    for name in ("m.json", "c.json", "f.json"):
        (tmp_path / name).write_text("{}")
    args = SimpleNamespace(secondary_cv_model=tmp_path / "m.json", secondary_cv_candidate_set=tmp_path / "c.json",
                           secondary_cv_feature_schema=tmp_path / "f.json", temperature_k=300.0)

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(IO, "load_pair_context", boom)
    rep = IO.run_epoch_slow_mode_reseed(fraction=0.25, epoch_dir=epoch, phase_dirs=[epoch], states=[],
                                        seed_bank_dir=bank, args=args)
    assert rep["status"] == "error" and "synthetic failure" in rep["error"]
    assert not (bank / X3.DIR_NAME).exists()


def test_runner_never_writes_into_a_foreign_conformer_library(tmp_path):
    lib = tmp_path / "genpept_library"
    lib.mkdir()
    (lib / X3.DIR_NAME).mkdir()                                  # would be deleted if treated as a bank
    epoch = tmp_path / "adaptive_production" / "epoch_000"
    epoch.mkdir(parents=True)
    args = SimpleNamespace(secondary_cv_model=None, temperature_k=300.0)
    rep = IO.run_epoch_slow_mode_reseed(fraction=0.25, epoch_dir=epoch, phase_dirs=[epoch], states=[],
                                        seed_bank_dir=lib, args=args)
    assert rep["status"] == "skipped" and (lib / X3.DIR_NAME).exists()


# ------------------------------------------------------------------ geometry and I/O twins

def test_vectorised_torsions_and_contacts_match_the_reference_implementations():
    from gareus.cv import nonlocal_contact_cv_from_positions_nm
    from gareus.tica import backbone_dihedral_features

    rng = np.random.default_rng(3)
    pos = rng.normal(0, 0.5, (20, 3))
    quads = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    ref = backbone_dihedral_features(pos, quads, [])
    assert np.allclose(IO.torsion_features(pos, quads)[0], ref, atol=1e-12)
    cargs = SimpleNamespace(contact_r0_a=5.0, contact_beta_a_inv=3.0, contact_normalize=True)
    pairs = [(0, 10, 1.0), (1, 15, 1.0), (2, 19, 0.5)]
    assert IO.contact_cv(pos, pairs, cargs)[0] == pytest.approx(nonlocal_contact_cv_from_positions_nm(pos, pairs, cargs))


def test_read_solute_positions_checks_names(tmp_path):
    p = tmp_path / "x.pdb"
    p.write_text(
        "ATOM      1  N   GLY A   1       1.000   2.000   3.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       4.000   5.000   6.000  1.00  0.00           C\n"
        "ATOM      3  O   HOH B   2       0.000   0.000   0.000  1.00  0.00           O\n")
    got = IO.read_solute_positions(p, [("GLY", "N"), ("GLY", "CA")])
    assert np.allclose(got, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert IO.read_solute_positions(p, [("GLY", "CA"), ("GLY", "N")]) is None


def test_frame_table_from_trajectories_joins_samples_and_maps_states(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from mdtraj.formats import XTCTrajectoryFile

    phase = tmp_path / "phase"
    (phase / "replica_trajectories").mkdir(parents=True)
    (phase / "samples" / "seg_001").mkdir(parents=True)
    rng = np.random.default_rng(5)
    rows = {"step": [], "replica": [], "window_id": [], "cv1": [], "cv2": []}
    for r in range(2):
        steps = np.arange(250, 250 * 41, 250)
        xyz = rng.normal(0, 0.3, (steps.size, 8, 3)).astype(np.float32)
        with XTCTrajectoryFile(str(phase / "replica_trajectories" / f"replica_{r:03d}.xtc"), "w") as fh:
            fh.write(xyz, time=steps * 0.0035, step=steps)
        for s in steps:
            rows["step"].append(int(s)); rows["replica"].append(r); rows["window_id"].append(r)
            rows["cv1"].append(0.1 * r); rows["cv2"].append(float(s))
    pq.write_table(pa.table({"step": pa.array(rows["step"], pa.uint64()), "replica": pa.array(rows["replica"], pa.uint16()),
                             "window_id": pa.array(rows["window_id"], pa.uint16()),
                             "cv1": pa.array(rows["cv1"], pa.float32()), "cv2": pa.array(rows["cv2"], pa.float32())}),
                   phase / "samples" / "seg_001" / "data.parquet")
    (phase / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,10\n1,11\n")
    ctx = SimpleNamespace(quads=[[0, 1, 2, 3], [4, 5, 6, 7]], solute_names=[("X", "Y")] * 8)
    t = IO.load_frame_table([phase], ctx, lag_ps=3.5, stride=2)
    assert t.X.shape == (40, 4) and sorted(set(t.state_id.tolist())) == [10, 11]
    assert np.all(t.cv2 == t.frame_index)                        # joined on (replica, step)
    assert t.lag_steps == 1000 and t.lag_ps == pytest.approx(3.5)
    assert np.unique(t.member).size == 2

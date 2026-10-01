"""Contact-map tICA CV1 fit (spec 2026-10-01-contact-map-cv1.md section 3; ``gareus.cv_selection.contact_map_cv1``)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.cv_selection import contact_map_cv1 as CM1


def test_rational_switch_is_the_6_12_form_without_the_pole():
    d = np.array([0.5, 2.0, 4.4, 4.6, 9.0])
    x = d / 4.5
    assert np.allclose(CM1.rational_switch(d, 4.5), (1 - x ** 6) / (1 - x ** 12))
    assert CM1.rational_switch(4.5, 4.5) == pytest.approx(0.5)


def test_lag_frames_are_positive_multiples_of_the_row_stride():
    assert CM1.lag_frames(50.0, 2.0, 1) == 25
    assert CM1.lag_frames(50.0, 2.0, 10) == 20
    assert CM1.lag_frames(5.0, 2.0, 10) == 10
    assert CM1.lag_frames(200.0, 2.0, 10) == 100


def _synthetic(n_members=40, n_frames=200, p=12, seed=0, slow=True):
    """Per member: a slow AR(1) latent (or none) and fast noise mapped into p pair distances."""
    rng = np.random.default_rng(seed)
    load = rng.normal(size=p)
    rows, members, frames, latent = [], [], [], []
    for m in range(n_members):
        x = rng.normal()
        for f in range(n_frames):
            x = (0.995 * x + 0.1 * rng.normal()) if slow else rng.normal()
            members.append(m); frames.append(f); latent.append(x)
            rows.append(4.5 + 1.2 * x * load + 0.6 * rng.normal(size=p))
    D = np.clip(np.asarray(rows), 2.0, None)
    latent = np.asarray(latent)
    n = D.shape[0]
    members, frames = np.asarray(members), np.asarray(frames)
    groups = np.asarray([f"seed{m // 2}" for m in members], dtype=object)
    basins = np.column_stack([(latent > q).astype(int) for q in (-0.5, 0.0, 0.5)])
    data = CM1.CV1Data(distances=D, member_ids=members, frame_index=frames, groups=groups,
                       cells=(np.arange(n) % 4), basin_labels=basins, frame_dt_ps=2.0,
                       ca_rows=np.arange(0, n, 5), ca_labels=(latent[::5] > 0).astype(int),
                       baselines={"noise": rng.normal(size=n)})
    return data, latent


def _definition(p):
    return {"schema": "swarm_contact_map_v1", "pairs": [{"i": k, "j": k + 3, "atoms_i": [k], "atoms_j": [k + 3]}
                                                        for k in range(p)], "sha256": "x"}


CFG = CM1.CV1FitConfig(tica_lag_ps=20.0, slowness_lag_ps=40.0, breadth_resamples=3, k_max_kcal=200.0)


def test_fit_tica_recovers_the_slow_direction_with_a_deterministic_sign():
    data, latent = _synthetic()
    S = CM1.rational_switch(data.distances)
    w = CM1.balanced_cell_weights(data.cells)
    vecs, ev, mu, sd = CM1.fit_tica(S, data.member_ids, data.frame_index, w, lag=10, n_modes=2, ridge=1e-6)
    z = (S @ vecs[0] - mu[0]) / sd[0]
    assert abs(np.corrcoef(z, latent)[0, 1]) > 0.95
    assert ev[0] > ev[1]
    assert vecs[0][np.argmax(np.abs(vecs[0]))] > 0                      # sign rule
    assert np.sum(w * z) == pytest.approx(0.0, abs=1e-10) and np.sum(w * z * z) == pytest.approx(1.0)
    again = CM1.fit_tica(S, data.member_ids, data.frame_index, w, lag=10, n_modes=2, ridge=1e-6)[0]
    assert np.array_equal(again, vecs)


def test_selection_picks_the_slow_mode_and_freezes_a_verifiable_model(tmp_path):
    data, latent = _synthetic()
    model, report = CM1.select_contact_map_cv1(data, CFG, definition=_definition(12), training={"n_rows": 1})
    assert report["status"] == CM1.STATUS_SELECTED and report["selected"] == "mode_1"
    row = report["candidates"]["mode_1"]
    assert row["passes"] and row["slowness_rho"] > 0.8 and min(row["half_split_abs_corr"]) > 0.9
    assert row["breadth_nats_mean"] > report["baselines"]["noise"]["breadth_nats_mean"]
    z = CM1.evaluate_model(model, data.distances)
    assert abs(np.corrcoef(z, latent)[0, 1]) > 0.95
    w = CM1.balanced_cell_weights(data.cells)
    assert np.sum(w * z) == pytest.approx(0.0, abs=1e-9) and np.sum(w * z * z) == pytest.approx(1.0)
    path = tmp_path / CM1.MODEL_NAME
    CM1.write_model(path, model)
    assert CM1.read_model(path) == model
    tampered = json.loads(path.read_text())
    tampered["weights"][0] += 1e-9
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest"):
        CM1.read_model(path)


def test_no_slow_mode_falls_back_to_contacts():
    data, _ = _synthetic(slow=False)
    model, report = CM1.select_contact_map_cv1(data, CFG, definition=_definition(12), training={})
    assert model is None and report["status"] == CM1.STATUS_FALLBACK
    assert report["fallback_cv1"] == "contacts"
    assert all(any("slowness" in r for r in c["reasons"]) for c in report["candidates"].values())


def test_resolvability_gate_refuses_a_mode_with_too_few_windows():
    data, _ = _synthetic()
    cfg = CM1.CV1FitConfig(tica_lag_ps=20.0, slowness_lag_ps=40.0, breadth_resamples=2, k_max_kcal=0.01)
    model, report = CM1.select_contact_map_cv1(data, cfg, definition=_definition(12), training={})
    assert model is None
    assert any("resolvable windows" in r for r in report["candidates"]["mode_1"]["reasons"])


def test_tie_set_then_slowest():
    rows = {"a": {"breadth_nats_mean": 0.50, "breadth_nats_sd": 0.05, "slowness_rho": 0.90},
            "b": {"breadth_nats_mean": 0.45, "breadth_nats_sd": 0.05, "slowness_rho": 0.95},
            "c": {"breadth_nats_mean": 0.10, "breadth_nats_sd": 0.05, "slowness_rho": 0.99}}
    assert CM1.breadth_tie_set(rows, 1.0) == {"a", "b"}
    assert CM1.breadth_tie_set(rows, 0.0) == {"a"}


def test_model_checks_refuse_wrong_schema_switch_and_width():
    model = CM1.build_model(_definition(3), CFG, [1.0, 2.0, 3.0], 0.5, {"mode": 1}, {})
    assert CM1.check_model(model) == model
    with pytest.raises(ValueError, match="one weight"):
        CM1.build_model(_definition(3), CFG, [1.0, 2.0], 0.5, {"mode": 1}, {})
    bad = {**model, "schema": "other"}
    with pytest.raises(ValueError, match="not a"):
        CM1.check_model(bad)


def test_cv1_data_rejects_misaligned_rows():
    data, _ = _synthetic(n_members=4, n_frames=20)
    with pytest.raises(ValueError, match="one entry per row"):
        CM1.CV1Data(**{**vars(data), "groups": data.groups[:-1]})
    with pytest.raises(ValueError, match="go together"):
        CM1.CV1Data(**{**vars(data), "ca_labels": None})


# --------------------------------------------------------------------------- swarm assembly

app = pytest.importorskip("openmm.app")


def _swarm_on_disk(tmp_path, n_members=3, n_rows=30, stride=5, record=(True, True, True)):
    """Tiny fake swarm: 10-residue topology, member PDB frames every ``stride`` rows."""
    from openmm import unit
    from gareus.swarm import contact_map as CMAP
    top = app.Topology()
    chain = top.addChain()
    C = app.Element.getBySymbol("C")
    for k in range(10):
        res = top.addResidue("ALA", chain, id=str(k + 1))
        for name in ("N", "CA", "C"):
            top.addAtom(name, C, res)
    tmp_path.mkdir(parents=True, exist_ok=True)
    topo_pdb = tmp_path / "topology.pdb"
    rng = np.random.default_rng(3)
    with open(topo_pdb, "w") as fh:
        app.PDBFile.writeFile(top, rng.normal(scale=0.8, size=(30, 3)) * unit.nanometer, fh)
    top = app.PDBFile(str(topo_pdb)).topology
    definitions = {a: CMAP.contact_map_definition(top, atoms=a) for a in (CMAP.ATOMS_HEAVY, CMAP.ATOMS_CA)}
    definition = definitions[CMAP.ATOMS_CA]                            # the map the fit uses by default
    evs = {a: CMAP.ContactMapEvaluator(d) for a, d in definitions.items()}
    rd = tmp_path / "round_000"
    frame_candidates, ok_traces, positions = [], {}, {}
    for m in range(n_members):
        md = rd / f"member_{m:04d}"
        (md / "frames").mkdir(parents=True)
        xyz = np.round(rng.normal(scale=0.8, size=(n_rows, 30, 3)), 3)      # nm, PDB precision
        ok_traces[m] = {"frame": np.arange(n_rows, dtype=float), "cv1": rng.random(n_rows),
                        "e2e_nm": rng.random(n_rows), "rg_nm": rng.random(n_rows)}
        if record[m]:
            for a, d in definitions.items():                           # members record both maps
                f_name, i_name = CMAP.file_names(a)
                np.save(md / f_name, np.vstack([evs[a](x) for x in xyz]))
                CMAP.write_index(md / i_name, d)
        for f in range(stride - 1, n_rows, stride):
            path = md / "frames" / f"frame_{f:05d}.pdb"
            with open(path, "w") as fh:
                app.PDBFile.writeFile(top, xyz[f] * unit.nanometer, fh)
            frame_candidates.append({"member_id": m, "frame": f, "pdb_path": str(path)})
        positions[m] = xyz
    from gareus.swarm.analyze import _feature_schema_from_index
    ca = [3 * k + 1 for k in range(10)]
    index = {"phi_torsions": [[ca[k] - 2, ca[k] - 1, ca[k], ca[k] + 1] for k in range(1, 10)],
             "psi_torsions": [[ca[k] - 1, ca[k], ca[k] + 1, ca[k] + 2] for k in range(9)]}
    schema = _feature_schema_from_index(index, "a" * 64)
    n = n_members * n_rows
    feats = rng.normal(size=(n, schema.width))
    dataset = SimpleNamespace(member_ids=np.repeat(np.arange(n_members), n_rows),
                              frame_index=np.tile(np.arange(n_rows), n_members),
                              groups=np.repeat(np.array(["s0", "s1", "s2"][:n_members], dtype=object), n_rows),
                              features=feats, shape_features=rng.random((n, 2)), feature_schema=schema,
                              frame_dt_ps=2.0)
    return dataset, ok_traces, frame_candidates, rd, topo_pdb, positions


def test_native_and_pdb_frame_sources_agree_on_the_pdb_rows(tmp_path):
    from gareus.swarm import contact_map_cv1_fit as F
    args = SimpleNamespace(swarm_contact_map_min_separation=3, swarm_contact_map_lambda_a=0.2)
    ds, tr, fc, rd, topo, _ = _swarm_on_disk(tmp_path / "a")
    warn = []
    native, d_nat, train_nat = F.assemble(ds, tr, fc, rd, topo, args, warn)
    assert train_nat["contact_map_source"] == F.SOURCE_TRACE and native.distances.shape == (90, 28) and not warn
    assert native.lag_stride == 1 and native.ca_rows.size == 18
    ds2, tr2, fc2, rd2, topo2, _ = _swarm_on_disk(tmp_path / "b", record=(True, False, True))
    pdb, d_pdb, train_pdb = F.assemble(ds2, tr2, fc2, rd2, topo2, args, warn)
    assert train_pdb["contact_map_source"] == F.SOURCE_PDB and pdb.lag_stride == 5 and warn
    assert pdb.distances.shape == (18, 28) and np.array_equal(pdb.frame_index % 5, np.full(18, 4))
    assert d_pdb["sha256"] == d_nat["sha256"]
    # same seed -> same coordinates: the PDB rows of the native map equal the PDB-frame map
    sel = np.flatnonzero(native.frame_index % 5 == 4)
    assert np.allclose(native.distances[sel], pdb.distances, atol=2e-3)
    assert set(pdb.baselines) == {"contacts", "e2e"}
    assert np.allclose(pdb.baselines["e2e"], 10.0 * np.concatenate(
        [tr2[m]["e2e_nm"][4::5] for m in range(3)]))


def test_basin_labels_pair_phi_and_psi_by_their_shared_ca(tmp_path):
    from gareus.adaptive.discovery_census import basin_codes
    from gareus.swarm.contact_map_cv1_fit import basin_labels
    ds, *_ = _swarm_on_disk(tmp_path)
    lab = basin_labels(ds.features, ds.feature_schema)
    assert lab.shape == (90, 8)                                   # residues 2-9 have both
    X = ds.features
    phi1 = -np.degrees(np.arctan2(X[:, 0], X[:, 1]))              # phi of residue 2 (CA index 4)
    psi1 = -np.degrees(np.arctan2(X[:, 18 + 2], X[:, 18 + 3]))    # psi block, second torsion: residue 2
    assert np.array_equal(lab[:, 0], basin_codes(phi1, psi1))


def _stale(an):
    (an / CM1.MODEL_NAME).write_text("{}")
    (an / CM1.REPORT_NAME).write_text('{"status": "selected"}')


def _gone(an):
    return not (an / CM1.MODEL_NAME).exists() and not (an / CM1.REPORT_NAME).exists()


def test_fit_hook_never_raises_and_removes_stale_artifacts(tmp_path):
    from gareus.swarm.analyze import _fit_contact_map_cv1
    from gareus.swarm.contact_map_cv1_fit import clear_artifacts
    _stale(tmp_path)
    clear_artifacts(tmp_path)                                       # what analyze runs on every path
    assert _gone(tmp_path)
    _stale(tmp_path)
    warn = []
    out = _fit_contact_map_cv1(tmp_path, None, {}, [], tmp_path, tmp_path / "missing.pdb", SimpleNamespace(), warn)
    assert out["status"] == "error" and warn
    _stale(tmp_path)
    ds, tr, fc, rd, topo, _ = _swarm_on_disk(tmp_path / "s")
    args = SimpleNamespace(swarm_contact_map_min_separation=3, swarm_contact_map_lambda_a=0.2, temperature_k=300.0)
    out = _fit_contact_map_cv1(tmp_path, ds, tr, fc, rd, topo, args, warn)
    # 3 seed families cannot fill 4 folds: recorded as an error, never raised, and neither the
    # stale model nor the stale report (which named it) survives
    assert out["status"] == "error" and "folds" in out["error"], out
    assert _gone(tmp_path)


def test_analyze_clears_cv1_artifacts_before_the_cv2_block():
    import inspect
    from gareus.swarm import analyze
    src = inspect.getsource(analyze.analyze_swarm_stage)
    assert src.index("_clear_cv1_artifacts(an)") < src.index('if secondary_cv_mode(args) == "auto":')


def test_recorded_map_with_a_wrong_row_count_falls_back_to_pdb_frames(tmp_path):
    from gareus.swarm import contact_map as CMAP
    from gareus.swarm import contact_map_cv1_fit as F
    args = SimpleNamespace(swarm_contact_map_min_separation=3, swarm_contact_map_lambda_a=0.2)
    for extra in (1, -1):
        ds, tr, fc, rd, topo, _ = _swarm_on_disk(tmp_path / f"x{extra}")
        path = rd / "member_0001" / CMAP.CA_FEATURES_NAME
        X = np.load(path)
        np.save(path, np.vstack([np.full((1, X.shape[1]), 9.9), X]) if extra > 0 else X[:-1])
        warn = []
        data, _d, training = F.assemble(ds, tr, fc, rd, topo, args, warn)
        assert training["contact_map_source"] == F.SOURCE_PDB
        assert any("not aligned" in w for w in warn) and not np.any(data.distances == 9.9)


def test_design_cells_do_not_depend_on_the_configured_cv1_kind(tmp_path):
    from gareus.swarm import contact_map_cv1_fit as F
    args = SimpleNamespace(swarm_contact_map_min_separation=3, swarm_contact_map_lambda_a=0.2)
    ds, tr, fc, rd, topo, _ = _swarm_on_disk(tmp_path)
    a = F.assemble(ds, tr, fc, rd, topo, args, [])[0]
    ds_rg_only = SimpleNamespace(**{**vars(ds), "shape_features": ds.shape_features[:, :1]})
    b = F.assemble(ds_rg_only, tr, fc, rd, topo, args, [])[0]
    assert np.array_equal(a.cells, b.cells)


def test_no_resampling_or_pick_slowest_puts_every_passing_mode_in_the_tie_set():
    from gareus.swarm.contact_map_cv1_fit import fit_config
    base = dict(temperature_k=300.0)
    assert fit_config(SimpleNamespace(**base)).use_breadth_tie
    assert not fit_config(SimpleNamespace(cv_selection_gain_resamples=0, **base)).use_breadth_tie
    assert not fit_config(SimpleNamespace(cv_selection_pick="slowest", **base)).use_breadth_tie
    rows = {"mode_2": {"mode": 2, "breadth_nats_mean": 0.5, "breadth_nats_sd": 0.0},
            "mode_1": {"mode": 1, "breadth_nats_mean": 0.5, "breadth_nats_sd": 0.0}}
    assert CM1.breadth_tie_set(rows, 0.0) == {"mode_1", "mode_2"}
    rows["mode_2"]["breadth_nats_mean"] = 0.49
    assert CM1.breadth_tie_set(rows, 0.0) == {"mode_1"}


def test_cli_and_yaml_keys(tmp_path):
    from gareus.cli import parse_args
    base = ["--seq", "GYDPETGTWG", "--out", str(tmp_path / "o")]
    a = parse_args(base)
    assert a.swarm_cv1_contact_map_fit is False and a.swarm_cv1_contact_map_r0_a is None
    assert a.swarm_cv1_contact_map_atoms == "ca"
    assert parse_args(base + ["--swarm-cv1-contact-map-fit"]).swarm_cv1_contact_map_fit is True
    with pytest.raises(SystemExit):
        parse_args(base + ["--swarm-cv1-contact-map-r0-a", "0"])
    cfg = tmp_path / "c.yaml"
    cfg.write_text("swarm:\n  cv1_contact_map_fit: true\n  cv1_contact_map_r0_a: 5.0\n")
    b = parse_args(["--config", str(cfg)] + base)
    assert b.swarm_cv1_contact_map_fit is True and b.swarm_cv1_contact_map_r0_a == 5.0

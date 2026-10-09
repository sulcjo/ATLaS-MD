import shutil
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd
import pytest

from aux_discovery_fixture import PDB, make_phase
from gareus.adaptive import aux_admission_io as H
from gareus.adaptive.aux_backfill import (BACKFILL_FILENAME, BackfillIncomplete, backfill_all,
                                          check_backfill_against_recorded, read_phase_backfill,
                                          write_phase_backfill)
from gareus.adaptive.aux_discovery.validation import ValidationStatus


def _model(seed=0):
    from gareus.adaptive.aux_discovery.descriptors import descriptor_definition
    from gareus.adaptive.aux_discovery.z3_search import Z3Candidate, emit_model
    t = md.load(str(PDB)); d = descriptor_definition(t.topology)
    rng = np.random.default_rng(seed)
    c = Z3Candidate(groups=(0, 1), C=0.1, w_std=rng.normal(size=36), mu=np.zeros(36), sd=np.ones(36),
                    z_raw_train_sd=1.0, info_gain=0.2, stability=0.9, corr_cv1=0.0, corr_cv2=0.0,
                    basin_gain=0.1, n_nonzero=36, passed=True, fail=[])
    return emit_model(c, d, t.topology.to_openmm(), label="t", provenance={})


def _z_ref(m):
    from gareus.auxiliary_cv.evaluate import z_from_positions
    # frames are stored in XTC (0.001 nm precision): the reference is the round-tripped geometry
    import tempfile
    xyz = md.load(str(PDB)).xyz
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.xtc"
        with md.formats.XTCTrajectoryFile(str(p), "w") as fh:
            fh.write(xyz, time=np.zeros(1), step=np.zeros(1, dtype=np.int32))
        with md.formats.XTCTrajectoryFile(str(p), "r") as fh:
            back = fh.read()[0]
    return float(np.ravel(z_from_positions(back[0].astype(np.float64), m))[0])


def test_backfill_complete_and_matches_positions(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300, 6300])}, {0: 0.0}, lambda s: 0.0)
    m = _model()
    info = write_phase_backfill(ph, m)
    df = read_phase_backfill(ph, m.model_sha256)
    assert info["n_samples"] == info["n_z"] == 3 and len(df) == 3
    assert np.allclose(df.aux_z, _z_ref(m), atol=1e-6)


def test_covers_every_lambda(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600]), "replica_1.xtc": (1, 1, [300, 600])},
                    {0: 0.0, 1: 0.5}, lambda s: 0.0)
    assert write_phase_backfill(ph, _model())["n_samples"] == 4


def test_missing_frames_refuse(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300])}, {0: 0.0}, lambda s: 0.0)
    f = ph / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f)
    extra = s.iloc[[0]].copy(); extra["step"] = 99300
    pd.concat([s, extra]).to_parquet(f)
    with pytest.raises(BackfillIncomplete) as e:
        write_phase_backfill(ph, _model())
    assert e.value.n_missing == 1 and e.value.examples == [[0, 99300]]
    assert not (ph / BACKFILL_FILENAME).exists()


def test_wrong_model_sha_refused(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    write_phase_backfill(ph, _model())
    with pytest.raises(ValueError, match="model"):
        read_phase_backfill(ph, "0" * 64)


def test_resume_overlap_later_file_wins(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600, 900]),
                                            "replica_0_resume_from_600.xtc": (0, 0, [600, 900, 1200])},
                    {0: 0.0}, lambda s: 0.0)
    assert write_phase_backfill(ph, _model())["n_samples"] == 4      # 300,600,900,1200 once each


def test_solute_atom_mismatch_refused(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    lines = (ph / "solute_only.pdb").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA")
    lines[i] = lines[i][:12] + " CB " + lines[i][16:]
    (ph / "solute_only.pdb").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="solute"):
        write_phase_backfill(ph, _model())


def test_backfill_all_respects_up_to_epoch(tmp_path):
    for n in ("epoch_000", "epoch_001", "epoch_002"):
        make_phase(tmp_path, n, {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    ad = tmp_path / "adaptive_production"
    out = backfill_all(ad, _model(), up_to_epoch=1)
    assert len(out) == 2
    assert (ad / "epoch_001" / BACKFILL_FILENAME).exists() and not (ad / "epoch_002" / BACKFILL_FILENAME).exists()


def test_check_against_recorded(tmp_path):
    m = _model()
    ph = make_phase(tmp_path, "epoch_001", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    f = ph / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f); s["aux_z_00"] = _z_ref(m); s.to_parquet(f)
    r = check_backfill_against_recorded(ph, m)
    assert r["n_compared"] == 2 and r["max_abs_dev"] < 1e-5 and r["ok"] and r["tol"] == 0.05
    s["aux_z_00"] = _z_ref(m) + 1e-2; s.to_parquet(f)          # XTC-scale deviation passes
    assert check_backfill_against_recorded(ph, m)["ok"]
    s["aux_z_00"] = _z_ref(m) + 0.1; s.to_parquet(f)
    r = check_backfill_against_recorded(ph, m)
    assert not r["ok"] and abs(r["max_abs_dev"] - 0.1) < 1e-4
    with pytest.raises(ValueError, match="differs"):
        check_backfill_against_recorded(ph, m, raise_on_fail=True)


def test_frame_atom_count_mismatch_refused(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    lines = (ph / "solute_only.pdb").read_text().splitlines()
    i = max(k for k, l in enumerate(lines) if l.startswith(("ATOM", "HETATM")))
    (ph / "solute_only.pdb").write_text("\n".join(lines[:i] + lines[i + 1:]) + "\n")
    with pytest.raises(ValueError, match="atoms per frame"):
        write_phase_backfill(ph, _model())


# hook: a backfill failure must unfreeze (model, partition, admission, backfills written in this attempt)
def test_hook_backfill_failure_removes_frozen_files_and_backfills(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gareus.adaptive_production import AdaptiveDecisionPolicy, WindowStateRegistry
    ad = tmp_path / "adaptive_production"
    make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    make_phase(tmp_path, "epoch_001", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    # epoch_001 gets a sample with no frame -> BackfillIncomplete after epoch_000's backfill was written
    f = ad / "epoch_001" / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f); x = s.iloc[[0]].copy(); x["step"] = 99300; pd.concat([s, x]).to_parquet(f)
    m = _model()
    monkeypatch.setattr(H, "check_validation_record", lambda *a, **k: ValidationStatus(True, "ok", 5.0))
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    res = H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)])
    res.model = m
    monkeypatch.setattr(H, "_discover", lambda **kw: res)
    r = WindowStateRegistry()
    for c in (0.0, 0.5):
        r.add_state(c, 10.0, 0.0, 2.0, gamd_lambda=0.0)
    args = SimpleNamespace(out=str(tmp_path), traj_interval=300, distance_output_interval=300, exchange_interval=3000,
                           timestep_fs=3.5, adaptive_production_aux_settings_override=False, temperature_k=300.0,
                           gamd_production_steps=60000)
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_001", epoch=1, registry=r, diagnostics={},
                                    actions=[], policy=AdaptiveDecisionPolicy(aux_discovery=True, aux_reserve_slots=4),
                                    args=args, out_dir=tmp_path, phase_dirs=[])
    import json
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "error" and "no trajectory frame" in rep["error"]
    assert not list(ad.rglob(BACKFILL_FILENAME))
    assert [p.name for p in ad.iterdir() if p.name.startswith(("aux_model", "aux_eval_partition", "aux_admission"))] == []


def test_hook_records_backfill_shas(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gareus.adaptive_production import AdaptiveDecisionPolicy, WindowStateRegistry
    import json
    ad = tmp_path / "adaptive_production"
    make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    make_phase(tmp_path, "epoch_001", {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    monkeypatch.setattr(H, "check_validation_record", lambda *a, **k: ValidationStatus(True, "ok", 5.0))
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    res = H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)])
    res.model.model_sha256 = "e" * 64
    monkeypatch.setattr(H, "_discover", lambda **kw: res)
    m = _model()
    monkeypatch.setattr(H, "_backfill", lambda a, e, model: __import__("gareus.adaptive.aux_backfill", fromlist=["x"]).backfill_all(a, m, up_to_epoch=e))
    r = WindowStateRegistry()
    for c in (0.0, 0.5):
        r.add_state(c, 10.0, 0.0, 2.0, gamd_lambda=0.0)
    args = SimpleNamespace(out=str(tmp_path), traj_interval=300, distance_output_interval=300, exchange_interval=3000,
                           timestep_fs=3.5, adaptive_production_aux_settings_override=False, temperature_k=300.0,
                           gamd_production_steps=60000)
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_001", epoch=1, registry=r, diagnostics={},
                                    actions=[], policy=AdaptiveDecisionPolicy(aux_discovery=True, aux_reserve_slots=4),
                                    args=args, out_dir=tmp_path, phase_dirs=[])
    assert len(out) == 1
    adm = json.loads((ad / "aux_admission.json").read_text())
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert len(adm["backfill"]) == 2 and rep["backfill"] == adm["backfill"] and all(len(b["sha256"]) == 64 for b in adm["backfill"])

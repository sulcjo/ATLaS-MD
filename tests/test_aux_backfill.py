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
    df = read_phase_backfill(ph, m.model_sha256, expected_sha256=info["sha256"])
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
    info = write_phase_backfill(ph, _model())
    with pytest.raises(ValueError, match="model"):
        read_phase_backfill(ph, "0" * 64, expected_sha256=info["sha256"])


def test_resume_overlap_later_file_wins(tmp_path):
    # crash case: the original file ran past the checkpoint (900) before the kill; the resume file (first frame
    # at checkpoint + interval, as npt_driver.register_reporter schedules it) supersedes that frame
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600, 900]),
                                            "replica_0_resume_from_600.xtc": (0, 0, [900, 1200])},
                    {0: 0.0}, lambda s: 0.0)
    assert write_phase_backfill(ph, _model())["n_samples"] == 4      # 300,600,900,1200 once each


def test_resume_step_frame_kept_c10_layout(tmp_path):
    # c10 epoch_000 replica 0 (C2): replica_000.xtc ends AT the resume step 269700, the resume file starts at
    # 270000 (= 269700 + interval) and a sample exists at 269700: its only frame is the earlier file's last one.
    ph = make_phase(tmp_path, "epoch_000", {"replica_000.xtc": (0, 0, [269100, 269400, 269700]),
                                            "replica_000_resume_from_000269700.xtc": (0, 0, [270000, 270300])},
                    {0: 0.0}, lambda s: 0.0)
    info = write_phase_backfill(ph, _model())
    df = read_phase_backfill(ph, _model().model_sha256, expected_sha256=info["sha256"])
    assert info["n_samples"] == info["n_z"] == 5
    assert sorted(df.step.tolist()) == [269100, 269400, 269700, 270000, 270300]


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


def test_backfill_and_recorded_check_on_a_gamd_phase(tmp_path):
    # Task 17: a self-calibrated GaMD phase's samples are calib_steps ahead of its XTC steps.
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300, 6300])}, {0: 0.0}, lambda s: 0.0,
                    sample_step_offset=10200)
    m = _model()
    info = write_phase_backfill(ph, m)
    df = read_phase_backfill(ph, m.model_sha256, expected_sha256=info["sha256"])
    assert info["n_samples"] == info["n_z"] == 3 and sorted(df.step.tolist()) == [10500, 13500, 16500]
    f = ph / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f); s["aux_z_00"] = _z_ref(m); s.to_parquet(f)
    chk = check_backfill_against_recorded(ph, m)
    assert chk["ok"] and chk["n_compared"] == 3


def _add_end_sample(ph, step, window=0, replica=0):
    f = ph / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f)
    extra = s[s.replica == replica].iloc[[0]].copy(); extra["step"] = step; extra["window_id"] = window
    pd.concat([s, extra]).to_parquet(f)


def _write_final_pdb(ph, replica, window, src=PDB, *, final_step=742):
    (ph / "final_pdbs").mkdir(exist_ok=True)
    shutil.copy(src, ph / "final_pdbs" / f"replica_{replica:03d}_window_{window:03d}.pdb")
    if final_step is not None:          # production's end-of-loop checkpoint: the phase's final production step
        _write_manifest(ph, final_step)


def _write_manifest(ph, absolute_step):
    import json
    (ph / "checkpoints").mkdir(exist_ok=True)
    (ph / "checkpoints" / "production_checkpoint_manifest.json").write_text(json.dumps(
        {"schema": "gareus_production_checkpoint_v1", "prod_done": int(absolute_step), "absolute_step": int(absolute_step)}))


def test_off_grid_end_of_phase_sample_takes_its_final_pdb(tmp_path):
    # c10 epoch_000: the pool's step total (357142) is off the 300-step frame grid, so the phase's last sample
    # has no XTC frame; production's final_pdbs/replica_RRR_window_WWW.pdb holds exactly that configuration.
    from gareus.auxiliary_cv.evaluate import z_from_positions
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 742)
    _write_final_pdb(ph, 0, 0)
    m = _model()
    info = write_phase_backfill(ph, m)
    df = read_phase_backfill(ph, m.model_sha256, expected_sha256=info["sha256"])
    assert info["n_samples"] == info["n_z"] == 3 and info["n_from_xtc"] == 2 and info["n_from_final_pdb"] == 1
    z_pdb = float(np.ravel(z_from_positions(md.load(str(PDB)).xyz[0].astype(np.float64), m))[0])
    assert df.set_index("step").aux_z[742] == pytest.approx(z_pdb, abs=1e-9)


def test_end_of_phase_sample_without_final_pdb_refused(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 742)
    with pytest.raises(BackfillIncomplete) as e:
        write_phase_backfill(ph, _model())
    assert e.value.examples == [[0, 742]]


def test_final_pdb_of_another_window_refused(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0, 1: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 742, window=0)
    _write_final_pdb(ph, 0, 1)                 # the replica's earlier stop held window 1: not this sample
    with pytest.raises(BackfillIncomplete):
        write_phase_backfill(ph, _model())


def test_missing_frame_before_the_end_is_never_patched_from_final_pdb(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 900])}, {0: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 600)                   # mid-phase gap: only the replica's LAST sample may use the PDB
    _write_final_pdb(ph, 0, 0, final_step=900)
    with pytest.raises(BackfillIncomplete) as e:
        write_phase_backfill(ph, _model())
    assert e.value.examples == [[0, 600]]


def test_final_pdb_with_wrong_solute_refused(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 742)
    lines = PDB.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(("ATOM", "HETATM")) and l[12:16].strip() == "CA")
    lines[i] = lines[i][:12] + " CB " + lines[i][16:]
    bad = tmp_path / "bad.pdb"; bad.write_text("\n".join(lines) + "\n")
    _write_final_pdb(ph, 0, 0, src=bad)
    with pytest.raises(ValueError, match="final_pdbs"):
        write_phase_backfill(ph, _model())


# Task 10 (F02): a final_pdbs frame is used only with step evidence tying it to the sample
def test_final_pdb_refused_when_step_is_not_the_phase_final_step(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 742)
    _write_final_pdb(ph, 0, 0, final_step=1042)    # stale final_pdbs: the phase ended (and wrote them) elsewhere
    with pytest.raises(BackfillIncomplete, match="final production step 1042") as e:
        write_phase_backfill(ph, _model())
    assert e.value.examples == [[0, 742]] and not (ph / BACKFILL_FILENAME).exists()


def test_final_pdb_refused_without_a_checkpoint_manifest(tmp_path):
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    _add_end_sample(ph, 742)
    _write_final_pdb(ph, 0, 0, final_step=None)
    with pytest.raises(BackfillIncomplete, match="checkpoint manifest"):
        write_phase_backfill(ph, _model())


def test_final_pdb_refused_when_step_not_after_the_replica_xtc(tmp_path):
    # replica 1's XTC runs to 900, but its last sample (742) is earlier: final_pdbs cannot be that configuration
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600]), "replica_1.xtc": (1, 0, [300, 900])},
                    {0: 0.0}, lambda s: 0.0)
    f = ph / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f)
    s = s[~((s.replica == 1) & (s.step == 900))]
    extra = s[s.replica == 1].iloc[[0]].copy(); extra["step"] = 742
    pd.concat([s, extra]).to_parquet(f)
    _write_final_pdb(ph, 1, 0, final_step=742)
    with pytest.raises(BackfillIncomplete, match="XTC"):
        write_phase_backfill(ph, _model())


def test_backfill_entries_carry_the_phase_label(tmp_path):
    for n in ("epoch_000", "epoch_001"):
        make_phase(tmp_path, n, {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    ad = tmp_path / "adaptive_production"
    out = backfill_all(ad, _model(), up_to_epoch=1)
    assert [b["label"] for b in out] == ["epoch_000", "epoch_001"]


def test_readmission_backfill_updates_the_admission_record(tmp_path):
    import hashlib
    import json
    from gareus.adaptive_production import WindowStateRegistry
    for n in ("epoch_000", "epoch_001", "epoch_002"):
        make_phase(tmp_path, n, {"replica_0.xtc": (0, 0, [300])}, {0: 0.0}, lambda s: 0.0)
    ad = tmp_path / "adaptive_production"
    m = _model()
    m.write(ad / "aux_model.json")
    first = backfill_all(ad, m, up_to_epoch=0)                     # the freeze at epoch 0
    (ad / "aux_admission.json").write_text(json.dumps({
        "schema": "atlas-aux-admission-v1", "epoch": 0, "model_sha256": m.model_sha256, "backfill": first,
        "workers": [{"parent_state_id": 0, "aux_center": 1.2, "aux_k_kcal_mol": 2.0, "placement_rank": 0,
                     "burnin_phase_epoch": 1}]}))
    reg = WindowStateRegistry()
    reg.add_state(0.0, 10.0, 0.0, 2.0, gamd_lambda=0.0)
    out = H._readmit_missing_workers(ad, ad / "epoch_002", 2, reg)  # epochs 1-2 ran without the model
    assert len(out) == 1
    adm = json.loads((ad / "aux_admission.json").read_text())
    by_label = {b["label"]: b["sha256"] for b in adm["backfill"]}
    assert sorted(by_label) == ["epoch_000", "epoch_001", "epoch_002"]
    for label, sha in by_label.items():
        assert sha == hashlib.sha256((ad / label / BACKFILL_FILENAME).read_bytes()).hexdigest(), label


def test_final_pdb_for_a_replica_without_xtc_frames_needs_only_the_manifest_step(tmp_path):
    # replica 1 wrote no XTC frame (its only sample is the off-grid end of the phase): no frame bound to check,
    # so the checkpoint manifest's final production step alone ties final_pdbs/ to the sample
    ph = make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 600])}, {0: 0.0}, lambda s: 0.0)
    f = ph / "samples" / "seg_000" / "data.parquet"
    s = pd.read_parquet(f)
    extra = s.iloc[[0]].copy(); extra["replica"] = 1; extra["step"] = 742
    pd.concat([s, extra]).to_parquet(f)
    _write_final_pdb(ph, 1, 0, final_step=742)
    _add_end_sample(ph, 742)
    _write_final_pdb(ph, 0, 0, final_step=742)
    info = write_phase_backfill(ph, _model())
    assert info["n_from_final_pdb"] == 2
    _write_manifest(ph, 1042)                      # same layout, final step elsewhere: refused
    with pytest.raises(BackfillIncomplete, match="final production step 1042"):
        write_phase_backfill(ph, _model())

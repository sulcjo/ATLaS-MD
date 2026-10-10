# tests/test_aux_loaders.py
import json

import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.evaluate import z_from_dihedrals
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import segment_aux_schemas
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.correctness._io import IntegrityError
from gareus.kernel_identity import AuxPoolingRefused, refuse_aux_snapshots, snapshot_has_aux

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2, blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))
RUNTIME = {"platform": "Reference", "precision": "double"}
KI = {"kernel_identity_version": "kernel_identity_v1", "exchange_energy_version": "state_bias_matrix_v3_aux",
      "aux_model_sha256": MODEL.model_sha256}
STAGE_B_ROWS = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0,
                 "aux_model_sha256": None, "aux_center": 0.0, "aux_k": 0.0},
                {"window_id": 1, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0,
                 "aux_model_sha256": MODEL.model_sha256, "aux_center": 0.5, "aux_k": 2.0}]


def _defn():
    return definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 2.0, 0.5, "auxiliary")],
                           MODEL.model_sha256), MODEL)


def _write_segment(run, seg, aux, rng, *, nan_rows=()):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(run / "samples" / seg, aux_schema=SCHEMA if aux else None,
                            aux_runtime=RUNTIME if aux else None)
    for i in range(6):
        theta = rng.uniform(-np.pi, np.pi, size=2)
        if i in nan_rows:
            theta[0] = np.nan
        kw = {}
        if aux:
            kw = dict(torsions=theta, aux_z=[float(z_from_dihedrals(theta[None, :], MODEL)[0])])
        w.write_sample(step=100 * i, replica=i % 2, window_id=i % 2, cv1=0.1, cv2=None, potential=0.0,
                       boost_total=None, boost_dihedral=None, boost_nonbonded=None, **kw)
    w.close()


def _run(tmp_path, *, historical=False, eligible=True, nan_rows=()):
    """historical: False | "legacy" (v2 kernel) | "stage_b" (aux kernel, no features) | "aux_no_snapshot"."""
    from gareus.store import SegmentRegistry, WindowSnapshot
    rng = np.random.default_rng(1)
    reg = SegmentRegistry(tmp_path)
    segs = []
    if historical:
        s0 = reg.open_segment("run", None, 1)
        _write_segment(tmp_path, s0, historical == "aux_no_snapshot", rng)
        reg.close_segment(s0, 600)
        if historical == "legacy":
            WindowSnapshot(tmp_path).snapshot(s0, [{"window_id": 0, "center1": 0.2, "k1": 10.0},
                                                   {"window_id": 1, "center1": 0.2, "k1": 10.0}], "contacts", None)
        elif historical == "stage_b":
            WindowSnapshot(tmp_path).snapshot(s0, STAGE_B_ROWS, "contacts", None, kernel_identity=KI)
        segs.append(s0)
    s1 = reg.open_segment("run", segs[-1] if segs else None, 1)
    _write_segment(tmp_path, s1, True, rng, nan_rows=nan_rows); reg.close_segment(s1, 1200)
    WindowSnapshot(tmp_path).snapshot(s1, [], "contacts", None, kernel_identity=KI, state_definition=_defn(),
                                      equilibrium_analysis_eligible=eligible,
                                      phase_kind="production" if eligible else "pilot")
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 300.0}))
    return segs + [s1]


def test_segment_schemas(tmp_path):
    s0, s1 = _run(tmp_path, historical="legacy")
    sch = segment_aux_schemas(tmp_path)
    assert sch[s0] is None and sch[s1] == SCHEMA


def test_snapshot_has_aux_sees_kernel_identity_and_stage_b_rows():
    assert snapshot_has_aux({"kernel_identity": {"exchange_energy_version": "state_bias_matrix_v3_aux"}})
    assert snapshot_has_aux({"windows": [{"window_id": 0, "aux_k": 0.0, "aux_model_sha256": None, "aux_center": 0.0}]})
    assert not snapshot_has_aux({"windows": [{"window_id": 0, "center1": 0.1, "k1": 1.0}]})
    assert not snapshot_has_aux([])


def test_detection_never_imports_the_aux_package_for_a_legacy_run(tmp_path):
    import subprocess
    import sys
    import textwrap
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.2, "k1": 1.0}], "contacts", None)
    code = textwrap.dedent(f"""
        import sys
        from gareus.kernel_identity import refuse_aux_snapshots, run_has_aux
        assert not run_has_aux({str(tmp_path)!r})
        refuse_aux_snapshots({str(tmp_path)!r}, where="x")
        assert "gareus.auxiliary_cv" not in sys.modules
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_load_parquet_evaluates_aux_for_every_row(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path)
    d = load_parquet(tmp_path)
    assert np.isfinite(d.u_nk).all() and np.all(d.u_nk[:, 1] >= d.u_nk[:, 0])
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_legacy_kernel_segment_fails_closed_even_with_the_opt_out(tmp_path):
    """H3: a v2 segment in an auxiliary campaign directory is refused, never silently excluded."""
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical="legacy")
    for kw in ({}, {"exclude_segments_without_aux_features": True}):
        with pytest.raises(IntegrityError, match=s0):
            load_parquet(tmp_path, **kw)


def test_snapshotless_segment_fails_closed_even_with_the_opt_out(tmp_path):
    """H2: a segment without a window snapshot in an auxiliary run cannot be analysed without repair."""
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical="aux_no_snapshot")
    for kw in ({}, {"exclude_segments_without_aux_features": True}):
        with pytest.raises(IntegrityError, match=s0):
            load_parquet(tmp_path, **kw)


def test_stage_b_segment_refused_unless_explicitly_excluded(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    s0, _s1 = _run(tmp_path, historical="stage_b")
    with pytest.raises(IntegrityError, match=s0):
        load_parquet(tmp_path)
    d = load_parquet(tmp_path, exclude_segments_without_aux_features=True)
    assert d.u_nk.shape[0] == 6
    assert any(s0 in note and "6" in note for note in d.meta["load_notes"])


def test_ineligible_aux_segment_refused_unless_allowed(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, eligible=False)
    with pytest.raises(IntegrityError, match="eligib"):
        load_parquet(tmp_path)
    d = load_parquet(tmp_path, allow_ineligible_aux_segments=True)
    assert any("engineering" in n for n in d.meta["load_notes"])


def test_nan_aux_rows_refused_or_excluded_with_report_never_silently(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, nan_rows=(1, 3))
    with pytest.raises(IntegrityError, match="Incomplete"):
        load_parquet(tmp_path)
    d = load_parquet(tmp_path, aux_on_incomplete="exclude_and_report", aux_time_block_steps=300)
    assert d.u_nk.shape[0] == 4
    assert d.meta["aux_exclusion_report"]["n_excluded"] == 2


def test_other_pooling_paths_refuse_aux(tmp_path):
    from gareus.mbar_analysis.loaders import load_csv, load_data, load_npz
    from gareus.query import export_analysis_arrays_npz
    _run(tmp_path)
    with pytest.raises(AuxPoolingRefused, match="fixed-state exporter"):
        refuse_aux_snapshots(tmp_path, where="adaptive union")
    with pytest.raises(AuxPoolingRefused, match="fixed-state exporter"):
        export_analysis_arrays_npz(tmp_path, beta=0.4)
    with pytest.raises(RuntimeError, match="auxiliary"):         # Stage B guard, hoisted (B3)
        load_npz(tmp_path)
    with pytest.raises(RuntimeError, match="auxiliary"):
        load_csv(tmp_path)
    (tmp_path / "samples.csv").write_text("step\n1\n")          # tempt auto-selection
    d = load_data(tmp_path, None, source="auto", no_augment=True)
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_load_data_default_path_works_for_aux_run_without_rounds(tmp_path):
    from gareus.mbar_analysis.loaders import load_data
    _run(tmp_path)
    d = load_data(tmp_path, None, source="auto")                 # default no_augment=False (board 4)
    assert d.meta["aux_models"] == [MODEL.model_sha256]


def test_load_data_refuses_aux_round_dir_and_skips_plain_round_dir(tmp_path):
    from gareus.mbar_analysis.loaders import load_data
    _run(tmp_path)
    rd = tmp_path / "adaptive_feedback_round_01"
    (rd / "analysis_chunks").mkdir(parents=True)
    (rd / "umbrella_windows.csv").write_text("window\n0\n")
    (rd / "analysis_chunks" / "chunk_000.npz").write_bytes(b"")
    d = load_data(tmp_path, None, source="auto")
    assert any("augmentation skipped" in n for n in d.meta.get("load_notes", []))
    _run(rd)                                                     # the round itself carries aux states
    with pytest.raises(AuxPoolingRefused, match="adaptive round augmentation"):
        load_data(tmp_path, None, source="auto")


def test_adaptive_branch_refuses_aux_phase_dirs(tmp_path):
    """H4: the adaptive branch of load_data (union NPZ, epoch CSV, union Parquet) refuses aux phases."""
    from gareus.mbar_analysis.loaders import load_data
    ap = tmp_path / "adaptive_production"
    (ap / "epoch_000").mkdir(parents=True)
    _run(ap / "epoch_000")
    (ap / "state_registry.csv").write_text("state_id\n0\n")
    with pytest.raises(RuntimeError, match="auxiliary"):
        load_data(tmp_path, None)


def test_union_builder_scans_pilot_dirs(tmp_path):
    from gareus.adaptive_production import build_union_state_mbar_inputs
    pilot = tmp_path / "pilot"; pilot.mkdir()
    _run(pilot)
    (tmp_path / "ap").mkdir()
    with pytest.raises(AuxPoolingRefused, match="fixed-state exporter"):
        build_union_state_mbar_inputs(tmp_path / "ap", registry=None, pilot_dirs=[pilot])


def test_refuse_is_silent_without_aux(tmp_path):
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.2, "k1": 1.0}], "contacts", None)
    refuse_aux_snapshots(tmp_path, where="x")


def test_origin_counts_follow_exclusion(tmp_path):
    from gareus.mbar_analysis.data import clean
    from gareus.mbar_analysis.loaders import load_parquet
    _run(tmp_path, nan_rows=(1, 3))   # two rows with NaN torsions
    d = clean(load_parquet(tmp_path, aux_on_incomplete="exclude_and_report", aux_time_block_steps=300))
    K = d.u_nk.shape[1]
    nk = np.bincount(d.window, minlength=K)
    assert int(nk.sum()) == d.u_nk.shape[0] == d.cv.size
    assert np.isfinite(d.u_nk).all()


def test_sample_window_ids_must_map_to_frozen_columns(tmp_path):
    """load_parquet indexes u_nk columns by window_id: an origin outside the frozen table is refused."""
    from gareus.auxiliary_cv.offline import pool_aux_segments
    from gareus.query import load_samples
    _run(tmp_path)
    samples = load_samples(tmp_path, include_ineligible=True)
    samples["window_id"] = np.asarray(samples["window_id"]).astype(np.int64)
    samples["window_id"][0] = 5
    with pytest.raises(IntegrityError, match="window_id"):
        pool_aux_segments(tmp_path, samples, 0.4, {})

import csv, io, types, tempfile, pathlib
import numpy as np


def test_window_state_carries_gamd_lambda_and_roundtrips():
    from gareus.adaptive_production import WindowState
    s = WindowState(state_id=3, primary_center=0.2, primary_k=500.0, gamd_lambda=0.25)
    d = s.to_dict()
    assert d["gamd_lambda"] == 0.25
    assert WindowState.from_dict(d).gamd_lambda == 0.25
    assert WindowState.from_dict({"state_id": 1, "primary_center": 0.1, "primary_k": 1.0}).gamd_lambda == 0.0
    assert WindowState.from_dict({"state_id": 1, "primary_center": 0.1, "primary_k": 1.0, "gamd_lambda": ""}).gamd_lambda == 0.0


def _write_csv(rows, header):
    d = pathlib.Path(tempfile.mkdtemp()); p = d / "windows.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); [w.writerow(r) for r in rows]
    return p


def _args():
    return types.SimpleNamespace(primary_cv="contacts", secondary_cv="none",
                                 explicit_2d_primary_center_column="primary_cv_center",
                                 explicit_2d_primary_k_column="primary_cv_k_kcal",
                                 explicit_2d_secondary_center_column="secondary_cv_center",
                                 explicit_2d_secondary_k_column="secondary_cv_k_kcal_mol",
                                 explicit_2d_primary_cv_mode_column="primary_cv_mode",
                                 explicit_2d_secondary_cv_mode_column="secondary_cv_mode",
                                 explicit_2d_window_schema="generic")


def test_explicit_csv_parses_gamd_lambda_column():
    from gareus.windows import load_explicit_2d_window_csv
    p = _write_csv([{"primary_cv_center": 0.1, "primary_cv_k_kcal": 100, "gamd_lambda": 0.0},
                    {"primary_cv_center": 0.1, "primary_cv_k_kcal": 100, "gamd_lambda": 0.5},
                    {"primary_cv_center": 0.3, "primary_cv_k_kcal": 100, "gamd_lambda": 1.0}],
                   ["primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"])
    centers, ks, sec_c, sec_k, meta, *_rest = load_explicit_2d_window_csv(_args(), p)
    assert list(meta["gamd_lambdas"]) == [0.0, 0.5, 1.0]
    assert [r["gamd_lambda"] for r in meta["normalized_rows"]] == [0.0, 0.5, 1.0]
    assert len(centers) == 3


def test_explicit_csv_without_gamd_lambda_defaults_to_zero():
    from gareus.windows import load_explicit_2d_window_csv
    p = _write_csv([{"primary_cv_center": 0.1, "primary_cv_k_kcal": 100}], ["primary_cv_center", "primary_cv_k_kcal"])
    _c, _k, _sc, _sk, meta, *_rest = load_explicit_2d_window_csv(_args(), p)
    assert list(meta["gamd_lambdas"]) == [0.0]


def test_explicit_csv_rejects_lambda_outside_unit_interval():
    from gareus.windows import load_explicit_2d_window_csv
    p = _write_csv([{"primary_cv_center": 0.1, "primary_cv_k_kcal": 100, "gamd_lambda": 1.5}],
                   ["primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"])
    try:
        load_explicit_2d_window_csv(_args(), p)
    except ValueError as exc:
        assert "gamd_lambda" in str(exc)
    else:
        raise AssertionError("λ outside [0, 1] must be rejected")


def test_window_assignment_rows_write_gamd_lambda():
    from gareus.production import window_assignment_rows
    rows = window_assignment_rows(np.array([0.1, 0.1]), [100.0, 100.0], 300.0, gamd_lambdas=[0.0, 1.0])
    assert [r["gamd_lambda"] for r in rows] == [0.0, 1.0]
    rows = window_assignment_rows(np.array([0.1]), [100.0], 300.0)
    assert rows[0]["gamd_lambda"] == 0.0


def test_registry_from_window_csv_carries_gamd_lambda():
    """Build the registry the way production does: window_assignment_rows ->
    write_window_assignment_csv (umbrella_windows.csv) -> registry_from_window_csv."""
    from gareus.production import window_assignment_rows, write_window_assignment_csv
    from gareus.adaptive_production import registry_from_window_csv

    rows = window_assignment_rows(np.array([0.1, 0.2, 0.3]), [100.0, 100.0, 100.0], 300.0,
                                   gamd_lambdas=[0.0, 0.4, 0.9])
    d = pathlib.Path(tempfile.mkdtemp()); p = d / "umbrella_windows.csv"
    write_window_assignment_csv(p, rows)
    reg = registry_from_window_csv(p)
    states = sorted(reg.all_states(), key=lambda s: s.state_id)
    assert [s.gamd_lambda for s in states] == [0.0, 0.4, 0.9]


def test_active_window_csv_round_trip_carries_gamd_lambda():
    """The per-epoch active-window CSV (write_active_window_csv) is what the adaptive
    driver feeds back in as the next epoch's --windows-2d-csv (via load_explicit_2d_window_csv).
    A state's gamd_lambda set at epoch N must not collapse to zero at epoch N+1."""
    from gareus.adaptive_production import WindowStateRegistry
    from gareus.windows import load_explicit_2d_window_csv

    reg = WindowStateRegistry()
    reg.add_state(primary_center=0.1, primary_k=100.0, gamd_lambda=0.0)
    reg.add_state(primary_center=0.2, primary_k=100.0, gamd_lambda=0.4)
    reg.add_state(primary_center=0.3, primary_k=100.0, gamd_lambda=0.9)

    d = pathlib.Path(tempfile.mkdtemp())
    csv_path = d / "windows_epoch_001.csv"
    reg.write_active_window_csv(csv_path)

    centers, ks, sec_c, sec_k, meta, *_rest = load_explicit_2d_window_csv(_args(), csv_path)
    assert list(meta["gamd_lambdas"]) == [0.0, 0.4, 0.9]


def test_state_registry_csv_carries_gamd_lambda():
    """state_registry.csv (write_state_csv) is a diagnostic dump of the full registry;
    it must not silently drop gamd_lambda either."""
    from gareus.adaptive_production import WindowStateRegistry
    import csv as _csv

    reg = WindowStateRegistry()
    reg.add_state(primary_center=0.1, primary_k=100.0, gamd_lambda=0.6)
    d = pathlib.Path(tempfile.mkdtemp())
    csv_path = d / "state_registry.csv"
    reg.write_state_csv(csv_path)
    with csv_path.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    assert float(rows[0]["gamd_lambda"]) == 0.6


def test_state_subset_window_csv_round_trip_carries_gamd_lambda():
    """write_state_subset_window_csv feeds seg_args.windows_2d_csv for an
    adaptive-production scheduled segment (structural twin of
    write_active_window_csv); its output must round-trip through
    load_explicit_2d_window_csv without zeroing lambda."""
    from gareus.adaptive_production import WindowStateRegistry, write_state_subset_window_csv
    from gareus.windows import load_explicit_2d_window_csv

    reg = WindowStateRegistry()
    s0 = reg.add_state(primary_center=0.1, primary_k=100.0, gamd_lambda=0.0)
    s1 = reg.add_state(primary_center=0.2, primary_k=100.0, gamd_lambda=0.5)
    s2 = reg.add_state(primary_center=0.3, primary_k=100.0, gamd_lambda=1.0)

    d = pathlib.Path(tempfile.mkdtemp())
    csv_path = d / "state_subset_windows.csv"
    write_state_subset_window_csv(reg, csv_path, [s0.state_id, s1.state_id, s2.state_id])

    centers, ks, sec_c, sec_k, meta, *_rest = load_explicit_2d_window_csv(_args(), csv_path)
    assert list(meta["gamd_lambdas"]) == [0.0, 0.5, 1.0]


def test_post_pull_drop_rewrites_umbrella_windows_csv_with_surviving_lambdas():
    """drop_bad_us_windows_and_rebuild rewrites umbrella_windows.csv (load-bearing
    for the legacy MBAR loader, not diagnostic); the surviving states' lambda
    must not be zeroed by that rewrite."""
    import types, csv as _csv
    from gareus.production import drop_bad_us_windows_and_rebuild

    d = pathlib.Path(tempfile.mkdtemp())
    args = types.SimpleNamespace(temperature_k=300.0)
    centers_a = [0.1, 0.2, 0.3, 0.4]
    k_list = [100.0, 100.0, 100.0, 100.0]
    centers_nm = np.array(centers_a)
    ks_kj_nm2 = np.array(k_list)
    window_metadata = {
        "normalized_rows": [
            {"window": 0, "gamd_lambda": 0.0},
            {"window": 1, "gamd_lambda": 0.3},
            {"window": 2, "gamd_lambda": 0.6},
            {"window": 3, "gamd_lambda": 0.9},
        ],
    }
    drop_bad_us_windows_and_rebuild(
        d, [1],
        centers_a, k_list, centers_nm, ks_kj_nm2,
        None, None, None,
        [None] * 4, [None] * 4,
        {}, window_metadata, args,
    )
    csv_path = d / "umbrella_windows.csv"
    with csv_path.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    assert [float(r["gamd_lambda"]) for r in rows] == [0.0, 0.6, 0.9]


def test_derive_state_gamd_lambdas_helper():
    from gareus.production import _derive_state_gamd_lambdas

    # normalized_rows present, aligned, every row carries gamd_lambda -> use them.
    wm = {"normalized_rows": [{"gamd_lambda": 0.1}, {"gamd_lambda": 0.2}]}
    assert _derive_state_gamd_lambdas(wm, 2) == [0.1, 0.2]

    # normalized_rows missing gamd_lambda on some row -> fall back to existing if aligned.
    wm2 = {"normalized_rows": [{"gamd_lambda": 0.1}, {"window": 1}]}
    assert _derive_state_gamd_lambdas(wm2, 2, existing=[0.4, 0.5]) == [0.4, 0.5]

    # normalized_rows absent entirely, no usable existing -> zeros.
    assert _derive_state_gamd_lambdas({}, 3) == [0.0, 0.0, 0.0]

    # normalized_rows length mismatch (stale/unrenumbered) -> ignore it, fall back.
    wm3 = {"normalized_rows": [{"gamd_lambda": 0.7}]}
    assert _derive_state_gamd_lambdas(wm3, 2, existing=[0.2, 0.3]) == [0.2, 0.3]
    assert _derive_state_gamd_lambdas(wm3, 2, existing=None) == [0.0, 0.0]

    # existing wrong length is never trusted either -> zeros.
    assert _derive_state_gamd_lambdas(None, 2, existing=[0.9]) == [0.0, 0.0]


def test_exchange_delta_between_rungs_is_the_boost_difference():
    """Two states with identical umbrellas, λ=0 and λ=1, holding replicas 0 and 1.
    Swapping them costs exactly boost(x0;λ=1) - boost(x1;λ=1) (the λ=0 terms are zero)."""
    from gareus.production import apply_window_swap
    from gareus.pep_gamd import pep_gamd_boost_matrix_kj, PepGamdEnvelope
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    lambdas = np.array([0.0, 1.0]); v_pep = np.array([10.0, 30.0]); v_dih = np.array([5.0, 7.0])
    umbrella_kj = np.zeros((2, 2))
    bias = umbrella_kj + pep_gamd_boost_matrix_kj(v_pep, v_dih, lambdas, env)
    assignments = np.array([0, 1]); replica_of_window = np.array([0, 1])
    out = apply_window_swap(bias, 1.0 / 2.494, assignments, replica_of_window, 0, 1, None, force_accept=True)
    expected = float(bias[1, 0] + bias[0, 1] - bias[0, 0] - bias[1, 1])
    assert abs(out.delta_kj - expected) < 1e-12
    assert out.accepted and list(assignments) == [1, 0]


def test_boost_matrix_term_is_zero_for_a_pure_umbrella_ladder():
    from gareus.pep_gamd import pep_gamd_boost_matrix_kj, PepGamdEnvelope
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    M = pep_gamd_boost_matrix_kj(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.zeros(4), env)
    assert M.shape == (4, 2) and np.all(M == 0.0)


def test_assemble_bias_matrices_kcal_kj_invariant_holds_with_boost():
    """bias_kj == 4.184 * bias_kcal must hold even when the λ-ladder boost is
    nonzero -- both unit columns must carry the SAME quantity (umbrella + boost)."""
    from gareus.production import assemble_bias_matrices
    distance_bias_kcal = np.array([[1.0, 2.0], [3.0, 4.0]])
    ss_bias_kcal = np.zeros((2, 2))
    boost_bias_kj = np.array([[0.0, 5.0], [10.0, 0.0]])
    bias_kcal, bias_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_kj)
    assert np.allclose(bias_kj, 4.184 * bias_kcal, atol=1e-12)
    umbrella_kj = 4.184 * (distance_bias_kcal + ss_bias_kcal)
    assert np.allclose(bias_kj, umbrella_kj + boost_bias_kj, atol=1e-12)


def test_parquet_sample_writer_stores_raw_channel_energies_and_lambda():
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    d = pathlib.Path(tempfile.mkdtemp())
    w = ParquetSampleWriter(d, flush_rows=10)
    w.write_sample(step=1, replica=0, window_id=0, cv1=0.1, cv2=None, potential=-5.0,
                   boost_total=1.0, boost_dihedral=0.5, boost_nonbonded=0.0,
                   v_pep=12.5, v_dih=3.25, gamd_lambda=0.5)
    w.write_sample(step=2, replica=0, window_id=0, cv1=0.1, cv2=None, potential=-5.0,
                   boost_total=0.0, boost_dihedral=0.0, boost_nonbonded=0.0)   # defaults: NaN, NaN, 0.0
    w.flush()
    t = pq.read_table(sorted(d.glob("chunk_*.parquet"))[0]).to_pydict()
    assert t["v_pep_kj_mol"][0] == 12.5 and t["v_dih_kj_mol"][0] == 3.25 and t["gamd_lambda"][0] == 0.5
    assert np.isnan(t["v_pep_kj_mol"][1]) and t["gamd_lambda"][1] == 0.0

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

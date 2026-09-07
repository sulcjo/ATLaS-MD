import csv, math, pathlib, tempfile, types
import numpy as np


def _env():
    from gareus.pep_gamd import PepGamdEnvelope
    # Total: V in [-100, 100], k0max 0.6; Dihedral: V in [0, 50], k0max 0.4
    return PepGamdEnvelope(100.0, -100.0, 100.0, 0.6, 50.0, 0.0, 50.0, 0.4)


def test_deltav_max_is_boost_at_lambda_one_and_nonnegative():
    from gareus.swarm.ladder_design import deltav_max_kj
    from gareus.pep_gamd import pep_gamd_boost_kj
    rng = np.random.default_rng(0)
    vp = rng.uniform(-90, 90, 500); vd = rng.uniform(5, 45, 500)
    dv = deltav_max_kj(vp, vd, _env())
    assert dv.shape == (500,) and np.all(dv >= 0)
    assert np.allclose(dv, np.asarray(pep_gamd_boost_kj(vp, vd, 1.0, _env())))


def test_design_lambda_ladder_starts_at_zero_ends_at_one_and_widens():
    from gareus.swarm.ladder_design import design_lambda_ladder
    rng = np.random.default_rng(1)
    dv = np.abs(rng.normal(30.0, 12.0, 5000))          # kJ/mol, σ ≈ 12 → βσ ≈ 4.8 at 300 K
    d = design_lambda_ladder(dv, 300.0, target_beta_sigma=1.0)
    lam = d["lambdas"]
    assert lam[0] == 0.0 and lam[-1] == 1.0 and all(np.diff(lam) > 0)
    assert 3 <= len(lam) <= 12
    steps = np.diff(lam)
    assert steps[-1] >= steps[0] * 0.99                  # spacing does not shrink toward λ = 1
    assert len(d["ess_per_rung"]) == len(lam)


def test_design_lambda_ladder_respects_rung_bounds_and_flags_extrapolation():
    from gareus.swarm.ladder_design import design_lambda_ladder
    dv = np.abs(np.random.default_rng(2).normal(200.0, 80.0, 40))   # tiny, wide sample → poor ESS
    d = design_lambda_ladder(dv, 300.0, target_beta_sigma=1.0, max_rungs=6, ess_floor=1000)
    assert len(d["lambdas"]) <= 6 and d["lambdas"][-1] == 1.0
    assert d["extrapolated_from_rung"] is not None


def test_cv1_centers_span_observed_range_with_n_windows():
    from gareus.swarm.ladder_design import cv1_centers_from_samples
    c = cv1_centers_from_samples(np.random.default_rng(3).beta(2, 5, 20000), n_windows=16)
    assert c.shape == (16,) and 0.0 <= c[0] < c[-1] <= 1.0 and np.all(np.diff(c) > 0)


def test_cv1_centers_refuse_to_exceed_library_q99():
    """r7 measured max heavy-CV1 = 0.069; a centre a pull cannot reach must never be written."""
    from gareus.swarm.ladder_design import cv1_centers_from_samples
    lib = np.random.default_rng(6).uniform(0.0, 0.069, 1970)
    swarm = np.random.default_rng(7).uniform(0.0, 0.065, 20000)
    c = cv1_centers_from_samples(swarm, n_windows=8, library_cv1=lib)
    assert c[-1] <= np.quantile(lib, 0.99)
    try:
        cv1_centers_from_samples(np.random.default_rng(8).uniform(0.0, 0.30, 20000), n_windows=8, library_cv1=lib)
    except ValueError as e:
        assert "q99" in str(e) or "library" in str(e)
    else:
        raise AssertionError("centres beyond library coverage must raise")


def test_fsf_floor_per_rung_reports_and_warns_on_top_rung():
    """Attempts 6-7: at k0_Total = 1 the unclamped FSF reaches 0 at Vmin and the run NaNs. Floor = 1 − λ·k0max."""
    from gareus.swarm.ladder_design import fsf_floor_per_rung
    from gareus.pep_gamd import PepGamdEnvelope
    env_k1 = PepGamdEnvelope(100.0, -100.0, 100.0, 1.0, 50.0, 0.0, 50.0, 0.4)
    r = fsf_floor_per_rung([0.0, 0.25, 0.5, 1.0], env_k1, warn_threshold=0.5)
    assert r["Total"] == [1.0, 0.75, 0.5, 0.0] and math.isclose(r["Dihedral"][-1], 0.6)
    assert r["top_rung_floor_total"] == 0.0 and r["warn"] is True and r["warn_threshold"] == 0.5
    env_k04 = PepGamdEnvelope(100.0, -100.0, 100.0, 0.4, 50.0, 0.0, 50.0, 0.4)
    assert fsf_floor_per_rung([0.0, 1.0], env_k04, warn_threshold=0.5)["warn"] is False


def test_window_sigma_against_coverage_range_matches_measured_numbers():
    from gareus.swarm.ladder_design import window_sigma_cv
    assert math.isclose(window_sigma_cv(250.0, 300.0), 0.049, abs_tol=0.002)
    assert math.isclose(window_sigma_cv(800.0, 300.0), 0.027, abs_tol=0.002)


def test_curvature_positive_in_a_well_and_zero_where_empty():
    from gareus.swarm.ladder_design import cv1_curvature_kcal
    cv1 = np.random.default_rng(4).normal(0.5, 0.05, 50000)      # Gaussian well: F'' = kT/σ² > 0
    centers = np.array([0.5, 0.95])
    f2 = cv1_curvature_kcal(cv1, centers, 300.0)
    kT = 0.0019872041 * 300.0
    assert math.isclose(f2[0], kT / 0.05 ** 2, rel_tol=0.35)
    assert f2[1] == 0.0


def test_force_constants_from_curvature_subtract_and_clamp():
    from gareus.swarm.ladder_design import cv1_force_constants_from_curvature
    centers = np.linspace(0.1, 0.85, 16)
    kT = 0.0019872041 * 300.0
    spacing = centers[1] - centers[0]
    k_flat = cv1_force_constants_from_curvature(centers, np.zeros(16), 300.0)
    assert math.isclose(k_flat[5], kT / (spacing / 1.5) ** 2, rel_tol=1e-6)
    k_curved = cv1_force_constants_from_curvature(centers, np.full(16, 30.0), 300.0)
    assert all(kc < kf for kc, kf in zip(k_curved, k_flat))
    k_huge = cv1_force_constants_from_curvature(centers, np.full(16, 1e6), 300.0, k_min_kcal=5.0)
    assert all(k == 5.0 for k in k_huge)
    assert all(k <= 1200.0 for k in cv1_force_constants_from_curvature(np.array([0.1, 0.101]), np.zeros(2), 300.0))


def _args():
    return types.SimpleNamespace(primary_cv="contacts", secondary_cv="none",
                                 explicit_2d_primary_center_column="primary_cv_center",
                                 explicit_2d_primary_k_column="primary_cv_k_kcal",
                                 explicit_2d_secondary_center_column="secondary_cv_center",
                                 explicit_2d_secondary_k_column="secondary_cv_k_kcal_mol",
                                 explicit_2d_primary_cv_mode_column="primary_cv_mode",
                                 explicit_2d_secondary_cv_mode_column="secondary_cv_mode",
                                 explicit_2d_window_schema="generic")


def test_windows_csv_is_full_cross_product_and_loads_with_core_reader():
    from gareus.swarm.ladder_design import write_ladder_windows_csv
    from gareus.windows import load_explicit_2d_window_csv
    p = pathlib.Path(tempfile.mkdtemp()) / "windows.csv"
    write_ladder_windows_csv(p, np.array([0.2, 0.4, 0.6]), [100.0, 110.0, 120.0], [0.0, 0.5, 1.0])
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 9 and rows[0].keys() >= {"window", "primary_cv_mode", "primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"}
    centers, ks, _sc, _sk, meta, *_ = load_explicit_2d_window_csv(_args(), p)
    assert len(centers) == 9 and sorted(set(meta["gamd_lambdas"])) == [0.0, 0.5, 1.0]

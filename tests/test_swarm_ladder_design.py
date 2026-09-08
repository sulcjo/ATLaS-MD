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


def test_cv1_centers_no_longer_enforces_a_library_quantile_veto():
    """REPLACES the retired test_cv1_centers_refuse_to_exceed_library_q99 (2026-09-08 fix):
    round-0 windows are seeded from the swarm's OWN frames, never pulled from the GENPEPT
    library, so capping centres against a library q99 was checking the wrong pool -- and it
    was inert anyway (centres = linspace(lo, hi, n) always ends at hi, so n_windows could
    never lower the top centre). library_cv1/library_q are removed from the signature
    entirely; a swarm whose top exceeds a library-like array's q99 must return centres,
    never raise."""
    from gareus.swarm.ladder_design import cv1_centers_from_samples
    lib = np.random.default_rng(6).uniform(0.0, 0.069, 1970)  # kept only to document the old cap value
    swarm = np.random.default_rng(8).uniform(0.0, 0.30, 20000)  # this used to raise against lib's q99
    c = cv1_centers_from_samples(swarm, n_windows=8)
    assert c.shape == (8,)
    assert c[-1] > np.quantile(lib, 0.99)
    try:
        cv1_centers_from_samples(swarm, n_windows=8, library_cv1=lib)
    except TypeError:
        pass
    else:
        raise AssertionError("library_cv1/library_q must be removed from the signature entirely")


def test_probe_centre_seed_support_returns_nearest_distance_per_centre():
    from gareus.swarm.ladder_design import probe_centre_seed_support
    centres = np.array([0.0, 0.5, 1.0])
    seed_cv1 = np.array([0.05, 0.4, 0.6, 1.2])
    gaps = probe_centre_seed_support(centres, seed_cv1)
    # hand-computed: nearest to 0.0 is 0.05 (0.05); nearest to 0.5 is 0.4 or 0.6 (0.1);
    # nearest to 1.0 is 1.2 (0.2)
    assert np.allclose(gaps, [0.05, 0.1, 0.2])


def test_autotune_cv1_upper_bound_accepts_hi_unchanged_when_seeds_are_dense_to_the_top():
    """The explicit regression for the real 2026-09-08 run: the swarm's own observed CV1
    spans 0..0.06 (its own frames ARE the seed-bank candidate pool), and a library-like
    array topping out at 0.042 (the OLD veto's cap, no longer even passed in) must not
    lower the bound or raise -- seed_cv1 dense across the whole quantile range means the
    initial hi is accepted on the first probe."""
    from gareus.swarm.ladder_design import autotune_cv1_upper_bound
    rng = np.random.default_rng(10)
    cv1 = rng.uniform(0.0, 0.06, 20000)
    seed_cv1 = rng.uniform(0.0, 0.06, 20000)  # the swarm's own frames, dense to the top
    library_like_cap = 0.042  # the old q99 cap value -- not passed to the new function at all
    result = autotune_cv1_upper_bound(cv1, seed_cv1, n_windows=8)
    assert result["autotuned"] is False
    assert math.isclose(result["hi"], result["hi_initial"])
    assert result["hi"] > library_like_cap  # NOT lowered to (or below) the old cap
    assert result["n_probes"] == 1
    assert np.max(result["nearest_seed_gap"]) <= result["tol"]


def test_autotune_cv1_upper_bound_walks_down_to_the_highest_supported_seed():
    """Seeds absent above 0.6: hi must drop to the highest seed-supported candidate
    (autofound from the observed seed values, never a fixed grid) and every centre in the
    accepted result is within tol."""
    from gareus.swarm.ladder_design import autotune_cv1_upper_bound
    cv1 = np.array([0.0, 0.3, 0.6, 0.9])
    seed_cv1 = np.concatenate([np.arange(0.0, 0.6 + 1e-9, 0.01), [0.6]])
    result = autotune_cv1_upper_bound(
        cv1, seed_cv1, n_windows=2, lo_q=0.0, hi_q=1.0, max_seed_gap_sigma=0.5,
    )
    assert result["autotuned"] is True
    assert math.isclose(result["hi_initial"], 0.9)
    assert math.isclose(result["hi"], 0.6, abs_tol=1e-9)
    assert result["hi"] < result["hi_initial"]
    assert np.max(result["nearest_seed_gap"]) <= result["tol"] + 1e-9
    assert result["n_probes"] >= 2


def test_autotune_cv1_upper_bound_raises_with_measured_gap_and_tol_when_nothing_is_supported():
    """No seed anywhere near the observed CV1 range -- every candidate (there are none to
    walk down to, since all seeds sit above hi_initial) leaves centres unsupported."""
    from gareus.swarm.ladder_design import autotune_cv1_upper_bound, window_sigma_cv
    cv1 = np.linspace(0.0, 1.0, 200)
    seed_cv1 = np.array([5.0, 6.0])  # far outside [lo, hi_initial]
    try:
        autotune_cv1_upper_bound(cv1, seed_cv1, n_windows=4)
    except ValueError as exc:
        msg = str(exc)
        tol = 0.5 * window_sigma_cv(1200.0, 300.0)
        assert "gap" in msg.lower()
        assert f"{tol:.4f}" in msg
    else:
        raise AssertionError("no supported candidate anywhere must raise ValueError")


def test_cv1_centers_from_samples_delegates_to_autotune_when_seed_cv1_given():
    from gareus.swarm.ladder_design import cv1_centers_from_samples
    rng = np.random.default_rng(11)
    swarm = rng.uniform(0.0, 0.06, 20000)
    seed = rng.uniform(0.0, 0.06, 20000)  # the swarm's own frames as the seed pool
    probe_out: dict = {}
    c = cv1_centers_from_samples(swarm, n_windows=8, seed_cv1=seed, probe_out=probe_out)
    assert c.shape == (8,)
    assert probe_out["autotuned"] is False
    assert math.isclose(probe_out["hi"], probe_out["hi_initial"])
    assert np.allclose(c, probe_out["centres"])


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


def test_n_resolvable_windows_matches_measured_r7_coverage():
    """r7's coverage_range=0.069 at k_max=1200, 300 K resolves only 2 windows, not spec S2's 16 ceiling."""
    from gareus.swarm.ladder_design import n_resolvable_windows
    n = n_resolvable_windows(0.069, 300.0)
    assert n == 2
    # a wider range resolves proportionally more
    assert n_resolvable_windows(0.69, 300.0) == 20


def test_fsf_floor_per_rung_handles_dihedral_only_envelope():
    """has_total=False (dihedral-only envelope): Total is not applicable, warn is driven by Dihedral."""
    from gareus.swarm.ladder_design import fsf_floor_per_rung
    import types
    env = types.SimpleNamespace(k0max_total=0.6, k0max_dih=1.0, has_total=False)
    r = fsf_floor_per_rung([0.0, 0.5, 1.0], env, warn_threshold=0.5)
    assert r["Total"] is None
    assert math.isclose(r["Dihedral"][-1], 0.0)
    assert r["top_rung_floor_total"] is None
    assert r["warn"] is True  # driven by the Dihedral channel's top-rung floor (0.0 < 0.5)


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


def test_upper_bound_probe_count_is_bounded_on_a_dense_seed_pool():
    """The walk down observed seed values must not scale with the pool size: a dense pool
    whose top is unsupported used to probe once per unique seed value (22k+ on a real run)."""
    from gareus.swarm.ladder_design import autotune_cv1_upper_bound, MAX_UPPER_BOUND_PROBES
    rng = np.random.default_rng(0)
    # Dense support up to 0.30, then a bare gap to a lone outlier that drags hi above it.
    seeds = np.concatenate([rng.uniform(0.0, 0.30, 20000), np.array([0.90])])
    cv1 = np.concatenate([seeds, np.full(400, 0.95)])  # q99.5 lands in the empty band
    out = autotune_cv1_upper_bound(cv1, seeds, n_windows=8, temperature_k=300.0,
                                   k_max_kcal=1200.0, max_seed_gap_sigma=0.05)
    assert out["autotuned"] is True
    assert out["hi"] < out["hi_initial"]
    assert out["n_probes"] <= MAX_UPPER_BOUND_PROBES + 1
    assert float(np.max(out["nearest_seed_gap"])) <= out["tol"]


def test_upper_bound_candidates_are_still_observed_seed_values():
    """Bounding the probe count must subsample the observed seeds, never invent a grid."""
    from gareus.swarm.ladder_design import autotune_cv1_upper_bound
    seeds = np.concatenate([np.linspace(0.0, 0.20, 500), np.array([0.80])])
    cv1 = np.concatenate([seeds, np.full(50, 0.85)])
    out = autotune_cv1_upper_bound(cv1, seeds, n_windows=4, temperature_k=300.0,
                                   k_max_kcal=1200.0, max_seed_gap_sigma=0.05)
    assert out["autotuned"] is True
    assert np.min(np.abs(seeds - out["hi"])) < 1e-12

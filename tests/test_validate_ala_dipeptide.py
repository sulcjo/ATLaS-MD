import numpy as np

from validate_ala_dipeptide import (
    BASINS,
    periodic_delta_deg,
    basin_metrics,
    compare,
)


def test_periodic_delta_wraps():
    d = periodic_delta_deg(np.array([170.0]), np.array([-170.0]))
    assert abs(abs(d[0]) - 20.0) < 1e-9  # 170 and -170 are 20 deg apart, not 340


def _grid_with_well(p0, q0, sampled_only_near=True):
    centers = np.linspace(-175, 175, 36)
    PHI, PSI = np.meshgrid(centers, centers, indexing="ij")
    pmf = (periodic_delta_deg(PHI, p0) ** 2 + periodic_delta_deg(PSI, q0) ** 2) / 400.0
    pmf = pmf - pmf.min()
    counts = np.zeros_like(pmf)
    if sampled_only_near:
        near = (np.abs(periodic_delta_deg(PHI, p0)) < 30) & (np.abs(periodic_delta_deg(PSI, q0)) < 30)
        counts[near] = 100.0
    else:
        counts[:] = 100.0
    return centers, pmf, counts


def test_basin_metrics_finds_well_and_flags_unsampled():
    centers, pmf, counts = _grid_with_well(-70.0, -30.0)  # alpha_R well, sampled only there
    m = basin_metrics(centers, centers, pmf, counts, BASINS, kbt_kcal=0.596)
    assert m["alpha_r"]["min_pmf_kcal"] < 0.5
    assert m["alpha_r"]["sampled"] is True
    assert m["c7ax"]["sampled"] is False  # no samples there


def test_basins_cover_expected_regions():
    assert set(BASINS) >= {"ppii", "beta", "alpha_r", "alpha_l", "c7ax"}
    assert abs(BASINS["alpha_r"][0] + 70) < 20 and abs(BASINS["alpha_r"][1] + 30) < 20


def test_compare_reports_pass_when_estimator_matches_reference():
    centers, pmf, counts = _grid_with_well(-70.0, 140.0, sampled_only_near=False)  # PPII, fully sampled
    ref = {"phi_deg": centers, "psi_deg": centers, "pmf_kcal_mol": pmf, "counts": counts}
    est = {"phi_deg": centers, "psi_deg": centers, "pmf_kcal_mol": pmf.copy(), "counts": counts.copy()}
    rep = compare(ref, {"gamd_cumulant2": est}, BASINS, kbt_kcal=0.596)
    e = rep["estimators"]["gamd_cumulant2"]
    assert e["passes_1kcal"] is True
    assert e["max_abs_delta_well_sampled_kcal"] < 1e-9


def test_compare_flags_disagreement():
    centers, pmf, counts = _grid_with_well(-70.0, 140.0, sampled_only_near=False)
    ref = {"phi_deg": centers, "psi_deg": centers, "pmf_kcal_mol": pmf, "counts": counts}
    # estimator with a 3 kcal/mol offset everywhere -> per-basin min differs by 3
    est = {"phi_deg": centers, "psi_deg": centers,
           "pmf_kcal_mol": pmf + 3.0 * (periodic_delta_deg(*np.meshgrid(centers, centers, indexing="ij")[:1], 0.0) * 0 + 1.0),
           "counts": counts.copy()}
    # add a real distortion: deepen a different basin so min locations diverge
    est["pmf_kcal_mol"] = pmf.copy()
    PHI, PSI = np.meshgrid(centers, centers, indexing="ij")
    est["pmf_kcal_mol"] += 2.0 * (np.abs(periodic_delta_deg(PHI, -70.0)) < 35) * (np.abs(periodic_delta_deg(PSI, 140.0)) < 35)
    rep = compare(ref, {"bad": est}, BASINS, kbt_kcal=0.596)
    assert rep["estimators"]["bad"]["passes_1kcal"] is False

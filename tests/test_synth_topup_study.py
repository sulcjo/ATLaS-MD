import numpy as np

from gareus.synth import LANDSCAPES
from gareus.synth.ess import tau_int
from gareus.synth.sampler import BOOST_A, Window, boost_dv, boost_reference_energy
from gareus.synth.topup_study import FROZEN_BUDGET_HOURS as H, run_arm, run_study, wins


def test_boost_is_a_lambda_scaled_flattening_below_the_reference_energy():
    ls = LANDSCAPES["gated-barrier"]
    x1, x2 = np.array([0.18, 0.5]), np.array([-0.65, 0.9])       # a deep well point and a ridge point
    f = ls.energy(x1, x2)
    zero = boost_dv(ls, Window(0.2, 50.0, 0.0, 50.0, lam=0.0), x1, x2)
    half = boost_dv(ls, Window(0.2, 50.0, 0.0, 50.0, lam=0.5), x1, x2)
    full = boost_dv(ls, Window(0.2, 50.0, 0.0, 50.0, lam=1.0), x1, x2)
    assert np.all(zero == 0.0)
    assert np.allclose(full, 2.0 * half)
    e_ref = boost_reference_energy(ls)
    assert f[0] < e_ref < f[1]
    assert np.isclose(full[0], BOOST_A * (e_ref - f[0])) and full[1] == 0.0   # only below E_ref is lifted


def test_tau_scales_with_the_number_of_exchange_partners():
    ls = LANDSCAPES["rugged-2d"]
    w = Window(0.3, 50.0, 0.2, 50.0)
    assert tau_int(ls, w) == tau_int(ls, w, n_partners=2)
    assert np.isclose(tau_int(ls, w, n_partners=1), tau_int(ls, w, n_partners=0))
    assert np.isclose(tau_int(ls, w, n_partners=4), 0.5 * tau_int(ls, w))


def test_arms_spend_equal_modelled_wall_hours():
    ls = LANDSCAPES["rugged-2d"]
    a = run_arm(ls, arm="uniform", seed=1, wall_hours_budget=H["rugged-2d"])
    b = run_arm(ls, arm="topup", seed=1, wall_hours_budget=H["rugged-2d"])
    assert abs(a.hours_used - b.hours_used) / a.hours_used < 0.02


def _arms(name, seeds=range(5)):
    ls = LANDSCAPES[name]
    a = [run_arm(ls, arm="uniform", seed=s, wall_hours_budget=H[name]) for s in seeds]
    b = [run_arm(ls, arm="topup", seed=s, wall_hours_budget=H[name]) for s in seeds]
    return a, b


def test_topups_beat_uniform_on_a_heterogeneous_landscape():
    a, b = _arms("gated-barrier")         # a win: B more than 1 % below A, on max sigma AND on PMF RMSE
    assert wins([r.max_sigma for r in b], [r.max_sigma for r in a]) >= 4
    assert wins([r.pmf_rmse for r in b], [r.pmf_rmse for r in a]) >= 4


def test_topups_match_uniform_on_a_homogeneous_landscape():
    a, b = _arms("harmonic-bowl")
    for key in ("max_sigma", "pmf_rmse"):
        ma, mb = np.median([getattr(r, key) for r in a]), np.median([getattr(r, key) for r in b])
        assert abs(mb - ma) / ma < 0.05, key
    assert max(r.topup_md_fraction for r in b) <= 0.3 + 1e-9


def test_a_missing_bridge_is_routed_and_an_intact_layout_is_not():
    ls = LANDSCAPES["gated-barrier"]
    cut = run_arm(ls, arm="topup", seed=2, wall_hours_budget=H["gated-barrier"], remove_bridge=True)
    whole = run_arm(ls, arm="topup", seed=2, wall_hours_budget=H["gated-barrier"])
    assert cut.structural_routed
    assert not whole.structural_routed


def test_study_writes_a_summary(tmp_path):
    summary = run_study(["rugged-2d"], n_seeds=2, out=tmp_path, budgets=H)
    assert (tmp_path / "topup_study.json").exists() and "rugged-2d" in summary

"""Confirms the MBAR solver family relocated from analyze_gareus_mbar.py
into gareus.mbar_analysis.solvers is a real delegation (same object), and
that the CLI-override block in analyze_gareus_mbar.py's parse_args() -- which
used to mutate analyze_gareus_mbar.py's own globals() to apply --sambar-*
flags -- was correctly retargeted to gareus.mbar_analysis.solvers' own
namespace. Before this retarget, the override would silently become a no-op
after relocation: solve_mbar_sambar/solve_mbar_sambar_warmstart read
SAMBAR_EPOCHS etc. as bare names resolved against the module they are
DEFINED in, not the module that happens to also import the same name."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar as agm
import gareus.mbar_analysis.solvers as solvers_mod


def test_solvers_module_is_importable():
    import gareus.mbar_analysis.solvers  # noqa: F401


@pytest.mark.parametrize("name", [
    "logsumexp", "logsumexp_axis1_finite", "logsumexp_axis0_finite",
    "norm_logw", "solve_mbar_numba", "solve_mbar_numba_anderson",
    "solve_mbar_sambar_warmstart", "solve_mbar_sambar", "solve_mbar_lbfgs",
    "solve_mbar", "overlap_matrix", "_subset_logw_from_global_fk",
])
def test_analyze_gareus_mbar_reexports_the_same_object(name):
    assert getattr(agm, name) is getattr(solvers_mod, name)


def test_solve_mbar_still_solves_a_real_two_window_problem():
    """Smoke-level numeric regression: the relocated dispatcher must still
    produce a converged, sane MBAR solve after the move."""
    rng = np.random.default_rng(0)
    beta = 0.5
    centers = [0.0, 1.0]
    k_spring = [30.0, 30.0]
    n_per_window = 3000
    cv_parts, win_parts = [], []
    for k, (c, ks) in enumerate(zip(centers, k_spring)):
        sigma = 1.0 / np.sqrt(beta * ks)
        cv_parts.append(rng.normal(c, sigma, n_per_window))
        win_parts.append(np.full(n_per_window, k, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)
    u_nk = beta * 0.5 * np.asarray(k_spring)[None, :] * (cv[:, None] - np.asarray(centers)[None, :]) ** 2

    res = solvers_mod.solve_mbar(u_nk, window, backend="numpy", tol=1e-10, maxiter=20000)

    assert res["converged"]
    assert np.all(np.isfinite(res["f_k"]))


# --- the CLI-override retarget regression ----------------------------------

def test_sambar_epochs_cli_override_updates_the_solvers_module_attribute():
    """Exercises the real, unmodified parse_args() code path end-to-end --
    argparse's `input` positional has no filesystem-existence type=
    validator (confirmed: `p.add_argument('input', help=...)`, no `type=`),
    so a nonexistent dummy path parses fine; parse_args() applies the
    override block as its last step before returning, so this reaches
    exactly the mechanism under test without needing to extract any new
    helper function out of parse_args()."""
    original = solvers_mod.SAMBAR_EPOCHS

    agm.parse_args(["dummy_run_dir", "--sambar-epochs", "7"])

    try:
        assert solvers_mod.SAMBAR_EPOCHS == 7
    finally:
        solvers_mod.SAMBAR_EPOCHS = original


def test_sambar_epochs_cli_override_actually_reaches_a_solver_call(monkeypatch):
    """The stronger check: prove the overridden value is what a real
    solve_mbar(backend='sambar') call receives, not just that the module
    attribute changed (a weaker check that would pass even if a later
    refactor stopped reading that global at all)."""
    received = {}

    def fake_warmstart(u_nk, window, epochs=None, **kwargs):
        received["epochs"] = epochs
        K = u_nk.shape[1]
        return np.zeros(K, dtype=np.float64)

    monkeypatch.setattr(solvers_mod, "solve_mbar_sambar_warmstart", fake_warmstart)
    monkeypatch.setattr(solvers_mod, "SAMBAR_EPOCHS", 7)
    monkeypatch.setattr(solvers_mod, "SAMBAR_POLISH_BACKEND", "numpy")

    u_nk = np.zeros((10, 2), dtype=np.float64)
    window = np.array([0] * 5 + [1] * 5, dtype=np.int64)

    solvers_mod.solve_mbar(u_nk, window, backend="sambar", maxiter=1)

    # solve_mbar_sambar (the wrapper solve_mbar dispatches to for
    # backend='sambar') passes its own `epochs` parameter through to
    # solve_mbar_sambar_warmstart -- when solve_mbar's own `sambar_epochs`
    # argument is left at its None default (the common case, matching the
    # main analyze() call site), solve_mbar_sambar's internal
    # `epochs = SAMBAR_EPOCHS if epochs is None else int(epochs)` resolves
    # the module-level SAMBAR_EPOCHS -- monkeypatched to 7 above -- before
    # ever calling solve_mbar_sambar_warmstart, so the warmstart call itself
    # already receives 7, not None.
    assert received["epochs"] == 7

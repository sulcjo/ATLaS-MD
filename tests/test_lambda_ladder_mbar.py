"""Task 6: MBAR reduced potentials carry the lambda-ladder boost term.

reconstruct_bias_matrix's window dicts use "center1"/"k1" (not "center"/"k") --
see its docstring; this test uses the real keys.
"""
import numpy as np


def test_reconstruct_bias_matrix_adds_boost_term_per_state_lambda():
    from gareus.query import reconstruct_bias_matrix
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    windows = [{"center1": 0.1, "k1": 100.0, "gamd_lambda": 0.0}, {"center1": 0.1, "k1": 100.0, "gamd_lambda": 1.0}]
    cv = np.array([0.1, 0.2]); v_pep = np.array([10.0, 30.0]); v_dih = np.array([5.0, 7.0]); beta = 1.0 / 2.494
    u = reconstruct_bias_matrix(cv, None, windows, beta, v_pep=v_pep, v_dih=v_dih, envelope=env)
    # Baseline computed from only the lambda=0 window: passing the full
    # `windows` list here (including the lambda=1.0 entry) without v_pep would
    # legitimately trip the "lambda>0 needs raw energies" guard tested below --
    # a single-window list containing only the lambda=0 rung is a safe way to
    # get the pure-umbrella comparison value (identical center1/k1 to window 1,
    # so it doubles as window 1's own umbrella-only term).
    u0 = reconstruct_bias_matrix(cv, None, [windows[0]], beta)
    assert np.allclose(u[:, 0], u0[:, 0])                                     # lambda = 0 state unchanged
    assert np.allclose(u[:, 1] - u0[:, 0], beta * np.array([pep_gamd_boost_kj(10.0, 5.0, 1.0, env), pep_gamd_boost_kj(30.0, 7.0, 1.0, env)]))


def test_reconstruct_bias_matrix_refuses_lambda_states_without_energies():
    from gareus.query import reconstruct_bias_matrix
    windows = [{"center1": 0.1, "k1": 100.0, "gamd_lambda": 0.5}]
    try:
        reconstruct_bias_matrix(np.array([0.1]), None, windows, 0.4)
    except ValueError as exc:
        assert "v_pep" in str(exc)
    else:
        raise AssertionError("a lambda>0 state without raw energies cannot be reweighted and must fail loudly")


def test_ladder_selects_the_exact_umbrella_only_path():
    from gareus.mbar_analysis.pmf import select_unbiased_method
    method, reason = select_unbiased_method(exp_ess=1.0, n_samples=10, gamd_ladder=True)
    assert method == 'umbrella_only' and 'ladder' in reason


def test_select_unbiased_method_default_unchanged_without_ladder():
    """gamd_ladder defaults to False; existing callers/behaviour untouched."""
    from gareus.mbar_analysis.pmf import select_unbiased_method
    method, reason = select_unbiased_method(1.0, 10)
    assert method != 'umbrella_only'

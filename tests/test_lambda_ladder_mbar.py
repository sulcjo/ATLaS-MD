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


def test_reconstruct_bias_matrix_propagates_nan_for_missing_energies():
    """2026-09-07 round-2 review, item A: reconstruct_bias_matrix now routes
    the ladder term through apply_ladder_boost_to_u, so a sample with a
    non-finite v_pep/v_dih must get NaN only in gamd_lambda>0 columns --
    never a silently fabricated 0.0 boost (the pre-fix behaviour of
    pep_gamd_boost_matrix_kj's own _channel_boost on a NaN input) -- and the
    gamd_lambda=0 column must stay genuinely zero-boost, untouched."""
    from gareus.query import reconstruct_bias_matrix
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    windows = [{"center1": 0.1, "k1": 100.0, "gamd_lambda": 0.0}, {"center1": 0.1, "k1": 100.0, "gamd_lambda": 1.0}]
    cv = np.array([0.1, 0.2])
    v_pep = np.array([10.0, np.nan])   # sample 1 missing its raw energy
    v_dih = np.array([5.0, 7.0])
    beta = 1.0 / 2.494
    u = reconstruct_bias_matrix(cv, None, windows, beta, v_pep=v_pep, v_dih=v_dih, envelope=env)
    u0 = reconstruct_bias_matrix(cv, None, [windows[0]], beta)  # pure-umbrella baseline, same center1/k1
    # Sample 0 (finite energies): both columns finite, lambda=1 column carries the closed-form boost.
    assert np.isfinite(u[0, 0]) and np.isfinite(u[0, 1])
    assert np.isclose(u[0, 1] - u0[0, 0], beta * pep_gamd_boost_kj(10.0, 5.0, 1.0, env))
    # Sample 1 (missing v_pep): lambda=0 column stays finite and equal to
    # the umbrella baseline (boost is unconditionally 0 there, never NaN);
    # lambda=1 column is NaN, not a fabricated 0.0.
    assert np.isclose(u[1, 0], u0[1, 0])
    assert np.isnan(u[1, 1])


def test_ladder_selects_the_exact_umbrella_only_path():
    from gareus.mbar_analysis.pmf import select_unbiased_method
    method, reason = select_unbiased_method(exp_ess=1.0, n_samples=10, gamd_ladder=True)
    assert method == 'umbrella_only' and 'ladder' in reason


def test_select_unbiased_method_default_unchanged_without_ladder():
    """gamd_ladder defaults to False; existing callers/behaviour untouched."""
    from gareus.mbar_analysis.pmf import select_unbiased_method
    method, reason = select_unbiased_method(1.0, 10)
    assert method != 'umbrella_only'


# ---------------------------------------------------------------------------
# 2026-09-07 review fixes: gareus.mbar_analysis.ladder.apply_ladder_boost_to_u
#
# N != K throughout (5 samples x 3 states) so a transposition bug would raise
# a shape-mismatch error (or silently broadcast wrong on a square matrix,
# which N==K could hide).
# ---------------------------------------------------------------------------

def _env():
    from gareus.pep_gamd import PepGamdEnvelope
    return PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)


def test_apply_ladder_boost_to_u_adds_term_matching_closed_form_nxk():
    from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
    from gareus.pep_gamd import pep_gamd_boost_matrix_kj
    env = _env()
    n, k = 5, 3
    u0 = np.arange(n * k, dtype=np.float64).reshape(n, k)
    v_pep = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    v_dih = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
    lambdas = np.array([0.0, 0.3, 0.7])
    beta = 1.0 / 2.494
    meta = {}
    u = apply_ladder_boost_to_u(u0, v_pep, v_dih, lambdas, env, beta, meta)
    assert u.shape == (n, k)
    expected_boost_nk = pep_gamd_boost_matrix_kj(v_pep, v_dih, lambdas, env).T
    assert expected_boost_nk.shape == (n, k)
    np.testing.assert_allclose(u, u0 + beta * expected_boost_nk)
    assert np.allclose(u[:, 0], u0[:, 0])  # lambda=0 column untouched
    assert meta == {'gamd_ladder': True, 'gamd_ladder_samples_without_raw_energies': 0}


def test_apply_ladder_boost_to_u_propagates_nan_for_missing_energies_nxk():
    from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
    env = _env()
    n, k = 5, 3
    u0 = np.zeros((n, k))
    v_pep = np.array([10.0, np.nan, 14.0, 16.0, np.nan])
    v_dih = np.array([5.0, 5.5, np.nan, 6.5, np.nan])       # sample 2 missing only v_dih
    lambdas = np.array([0.0, 0.3, 0.7])
    meta = {}
    u = apply_ladder_boost_to_u(u0, v_pep, v_dih, lambdas, env, 1.0, meta)
    missing = [1, 2, 4]   # samples 1, 2, 4 each lack at least one of v_pep/v_dih
    for i in missing:
        assert np.isnan(u[i, 1]) and np.isnan(u[i, 2])   # every lambda>0 column
        assert u[i, 0] == 0.0                             # lambda=0 column: never fabricated, never NaN
    for i in (0, 3):
        assert np.isfinite(u[i, 1]) and np.isfinite(u[i, 2])
    assert meta['gamd_ladder_samples_without_raw_energies'] == len(missing)


def test_apply_ladder_boost_to_u_all_missing_raises_naming_v_pep_nxk():
    from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
    env = _env()
    u0 = np.zeros((5, 3))
    try:
        apply_ladder_boost_to_u(u0, np.full(5, np.nan), np.full(5, np.nan), np.array([0.0, 1.0, 0.0]), env, 1.0, {})
    except ValueError as exc:
        assert "v_pep" in str(exc)
    else:
        raise AssertionError("every sample missing raw energies while a state has lambda>0 must raise")


def test_apply_ladder_boost_to_u_no_envelope_raises_naming_v_pep_nxk():
    from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
    u0 = np.zeros((5, 3))
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        apply_ladder_boost_to_u(u0, v, v, np.array([0.0, 1.0, 0.0]), None, 1.0, {})
    except ValueError as exc:
        assert "v_pep" in str(exc)
    else:
        raise AssertionError("a state with lambda>0 and no envelope must raise")


def test_apply_ladder_boost_to_u_passthrough_when_no_lambda_active_nxk():
    from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
    u0 = np.arange(15.0).reshape(5, 3)
    meta = {}
    u = apply_ladder_boost_to_u(u0, np.full(5, np.nan), np.full(5, np.nan), np.zeros(3), None, 1.0, meta)
    assert u is u0   # true no-op, not merely numerically equal
    assert meta == {'gamd_ladder': False, 'gamd_ladder_samples_without_raw_energies': 0}


# ---------------------------------------------------------------------------
# load_pep_gamd_envelope: tries the single-run path, then the adaptive-union
# campaign-export path.
# ---------------------------------------------------------------------------

def test_load_pep_gamd_envelope_single_run_convention(tmp_path):
    import json
    from gareus.mbar_analysis.ladder import load_pep_gamd_envelope
    (tmp_path / 'shared_gamd_setup_globals.json').write_text(json.dumps({
        'all_globals': {'k0_Total': 0.8, 'Vmax_Total': 50.0, 'Vmin_Total': -50.0, 'threshold_energy_Total': 50.0,
                         'k0_Dihedral': 0.6, 'Vmax_Dihedral': 50.0, 'Vmin_Dihedral': -50.0, 'threshold_energy_Dihedral': 50.0}
    }))
    env = load_pep_gamd_envelope(tmp_path)
    assert env is not None and env.k0max_total == 0.8


def test_load_pep_gamd_envelope_adaptive_union_convention(tmp_path):
    import json
    from gareus.mbar_analysis.ladder import load_pep_gamd_envelope
    d = tmp_path / 'global_shared_gamd_setup'
    d.mkdir()
    (d / 'shared_gamd_setup_globals.json').write_text(json.dumps({
        'all_globals': {'k0_Total': 0.8, 'Vmax_Total': 50.0, 'Vmin_Total': -50.0, 'threshold_energy_Total': 50.0,
                         'k0_Dihedral': 0.6, 'Vmax_Dihedral': 50.0, 'Vmin_Dihedral': -50.0, 'threshold_energy_Dihedral': 50.0}
    }))
    env = load_pep_gamd_envelope(tmp_path)
    assert env is not None and env.k0max_total == 0.8


def test_load_pep_gamd_envelope_returns_none_when_absent(tmp_path):
    from gareus.mbar_analysis.ladder import load_pep_gamd_envelope
    assert load_pep_gamd_envelope(tmp_path) is None


# ---------------------------------------------------------------------------
# load_csv: R1 -- the has_vectors branch must NOT add the term a second time
# (Critical 1); the reconstructed branch must add it exactly once.
# ---------------------------------------------------------------------------

def _write_envelope(prod, k0_total=0.8, k0_dih=0.6):
    import json
    (prod / 'shared_gamd_setup_globals.json').write_text(json.dumps({
        'all_globals': {'k0_Total': k0_total, 'Vmax_Total': 50.0, 'Vmin_Total': -50.0, 'threshold_energy_Total': 50.0,
                         'k0_Dihedral': k0_dih, 'Vmax_Dihedral': 50.0, 'Vmin_Dihedral': -50.0, 'threshold_energy_Dihedral': 50.0}
    }))


def test_load_csv_has_vectors_branch_does_not_double_count_boost(tmp_path):
    """Critical 1: samples.csv carries pre-computed all-window bias vectors
    that, for a ladder run, already include the boost (production.py's
    assemble_bias_matrices folds boost_bias_kj in before the per-window
    vectors are sliced out) -- load_csv must not add it a second time."""
    import csv
    from gareus.mbar_analysis.loaders import load_csv
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    beta = 1.0 / 2.494
    centers = [0.2, 0.5]; ks = [10.0, 10.0]; lambdas = [0.0, 1.0]
    v_pep, v_dih = 10.0, 5.0
    # umbrella-only + (for the lambda=1 state) the boost already folded in,
    # exactly like production.py's real bias_matrix_kj/reduced_bias_matrix.
    boost_kj_per_state = [pep_gamd_boost_kj(v_pep, v_dih, lam, env) for lam in lambdas]
    with (tmp_path / 'umbrella_windows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['center_A', 'k_kcal_mol_A2'])
        w.writeheader()
        for c, k in zip(centers, ks):
            w.writerow({'center_A': c, 'k_kcal_mol_A2': k})
    with (tmp_path / 'samples.csv').open('w', newline='') as f:
        fieldnames = ['cv_A', 'window', 'replica', 'step', 'v_pep_kj_mol', 'v_dih_kj_mol', 'gamd_lambda',
                      'umbrella_reduced_bias_all_windows_json']
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        cv_val = 0.5
        umbrella_kcal = [0.5 * k * (cv_val - c) ** 2 for c, k in zip(centers, ks)]
        umbrella_kj = [4.184 * x for x in umbrella_kcal]
        total_kj = [u + b for u, b in zip(umbrella_kj, boost_kj_per_state)]
        reduced = [beta * x for x in total_kj]
        import json as _json
        wtr.writerow({'cv_A': cv_val, 'window': 1, 'replica': 0, 'step': 0,
                      'v_pep_kj_mol': v_pep, 'v_dih_kj_mol': v_dih, 'gamd_lambda': 1.0,
                      'umbrella_reduced_bias_all_windows_json': _json.dumps(reduced)})
    _write_envelope(tmp_path)
    (tmp_path / 'gareus_metadata.json').write_text('{"temperature_K": 300.0}')

    d = load_csv(tmp_path)
    assert d.meta.get('gamd_ladder') is True
    np.testing.assert_allclose(d.u_nk[0], reduced, rtol=1e-8)  # NOT reduced + beta*boost again


def test_load_csv_reconstructed_branch_adds_boost_once(tmp_path):
    """No all-window vectors in samples.csv -> load_csv reconstructs u from
    umbrella_windows.csv and must add the ladder term exactly once."""
    import csv
    from gareus.mbar_analysis.loaders import load_csv
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    with (tmp_path / 'umbrella_windows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['center_A', 'k_kcal_mol_A2'])
        w.writeheader()
        w.writerow({'center_A': 0.2, 'k_kcal_mol_A2': 10.0})
        w.writerow({'center_A': 0.5, 'k_kcal_mol_A2': 10.0})
    with (tmp_path / 'samples.csv').open('w', newline='') as f:
        fieldnames = ['cv_A', 'window', 'replica', 'step', 'v_pep_kj_mol', 'v_dih_kj_mol', 'gamd_lambda']
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        wtr.writerow({'cv_A': 0.5, 'window': 1, 'replica': 0, 'step': 0,
                      'v_pep_kj_mol': 10.0, 'v_dih_kj_mol': 5.0, 'gamd_lambda': 1.0})
    _write_envelope(tmp_path)
    (tmp_path / 'gareus_metadata.json').write_text('{"temperature_K": 300.0}')

    d = load_csv(tmp_path)
    assert d.meta.get('gamd_ladder') is True
    beta = d.beta
    umbrella_kj = 4.184 * 0.5 * 10.0 * (0.5 - 0.5) ** 2
    boost_kj = pep_gamd_boost_kj(10.0, 5.0, 1.0, env)
    expected = beta * (umbrella_kj + boost_kj)
    assert np.isclose(d.u_nk[0, 1], expected)


# ---------------------------------------------------------------------------
# load_parquet_adaptive_union: Important 4 -- the dominant adaptive routing
# path was completely untouched; must now embed the boost and set the flag.
# ---------------------------------------------------------------------------

def test_load_parquet_adaptive_union_embeds_ladder_boost_and_sets_flag(tmp_path):
    import csv as _csv
    import json as _json
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union

    adaptive_dir = tmp_path / 'adaptive_production'
    adaptive_dir.mkdir(parents=True)

    def _write_epoch(epoch_dir, samples, state_id, primary_center, primary_k=10.0):
        epoch_dir.mkdir(parents=True, exist_ok=True)
        reg = SegmentRegistry(epoch_dir)
        seg_id = reg.open_segment('run_001', None, 1)
        WindowSnapshot(epoch_dir).snapshot(
            seg_id, [{'window_id': 0, 'center1': primary_center, 'k1': primary_k}],
            cv1_type='distance', cv2_type=None,
        )
        writer = ParquetSampleWriter(epoch_dir / 'samples' / seg_id, flush_rows=1000)
        max_step = 0
        for step, replica, cv1, v_pep, v_dih, lam in samples:
            writer.write_sample(step, replica, 0, cv1, None, -100.0, 0.0, 0.0, 0.0,
                                 v_pep=v_pep, v_dih=v_dih, gamd_lambda=lam)
            max_step = max(max_step, step)
        writer.close()
        reg.close_segment(seg_id, end_step=max_step)
        (epoch_dir / 'epoch_window_map.csv').write_text(
            'epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n'
            f'0,{state_id},{primary_center},{primary_k},,\n', encoding='utf-8',
        )

    samples0 = [(i * 50, i % 3, 0.02 * i, 10.0 + 0.1 * i, 5.0 + 0.05 * i, 0.0) for i in range(20)]
    samples1 = [(i * 50, i % 3, 0.5 + 0.02 * i, 10.0 + 0.1 * i, 5.0 + 0.05 * i, 1.0) for i in range(20)]
    _write_epoch(adaptive_dir / 'epoch_000', samples0, state_id=0, primary_center=0.0)
    _write_epoch(adaptive_dir / 'epoch_001', samples1, state_id=1, primary_center=0.5)

    with (adaptive_dir / 'state_registry.csv').open('w', newline='') as f:
        wtr = _csv.DictWriter(f, fieldnames=[
            'state_id', 'active', 'created_epoch', 'retired_epoch', 'parent_state_id',
            'primary_center', 'primary_k', 'secondary_center', 'secondary_k',
            'usable_for_mbar', 'burnin_steps', 'gamd_lambda',
        ])
        wtr.writeheader()
        for row in (
            {'state_id': 0, 'primary_center': 0.0, 'primary_k': 10.0, 'secondary_center': '', 'secondary_k': '', 'gamd_lambda': 0.0},
            {'state_id': 1, 'primary_center': 0.5, 'primary_k': 10.0, 'secondary_center': '', 'secondary_k': '', 'gamd_lambda': 1.0},
        ):
            wtr.writerow({'active': 'True', 'created_epoch': 0, 'retired_epoch': '', 'parent_state_id': '',
                          'usable_for_mbar': 'True', 'burnin_steps': 0, **row})

    env_dir = adaptive_dir / 'global_shared_gamd_setup'
    env_dir.mkdir(parents=True)
    (env_dir / 'shared_gamd_setup_globals.json').write_text(_json.dumps({
        'all_globals': {'k0_Total': 0.8, 'Vmax_Total': 50.0, 'Vmin_Total': -50.0, 'threshold_energy_Total': 50.0,
                         'k0_Dihedral': 0.6, 'Vmax_Dihedral': 50.0, 'Vmin_Dihedral': -50.0, 'threshold_energy_Dihedral': 50.0}
    }))

    data = load_parquet_adaptive_union(adaptive_dir)
    assert data.meta.get('gamd_ladder') is True
    assert np.isfinite(data.u_nk).all()
    assert np.allclose(data.state_lambdas, [0.0, 1.0])
    # state 0 (lambda=0) column must equal a pure-umbrella reconstruction;
    # state 1 (lambda=1) column must differ from it (boost genuinely embedded).
    from gareus.query import reconstruct_bias_matrix
    windows_plain = [{'center1': 0.0, 'k1': 10.0}, {'center1': 0.5, 'k1': 10.0}]
    u_umbrella = reconstruct_bias_matrix(data.cv, None, windows_plain, data.beta)
    assert np.allclose(data.u_nk[:, 0], u_umbrella[:, 0])
    assert not np.allclose(data.u_nk[:, 1], u_umbrella[:, 1])

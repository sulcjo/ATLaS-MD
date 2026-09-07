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


# ---------------------------------------------------------------------------
# 2026-09-07 round-2 review, item C: load_npz (analysis_arrays.npz route)
# is a sixth u-building path -- must be wired the same way when the npz
# carries the ladder columns (Task 5), and must be a true no-op otherwise.
# ---------------------------------------------------------------------------

def test_load_npz_embeds_ladder_boost_when_arrays_present(tmp_path):
    import json
    from gareus.mbar_analysis.loaders import load_npz
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

    from gareus.units import K_B_KJ_PER_MOL_K
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    beta = 1.0 / (K_B_KJ_PER_MOL_K * 300.0)   # must match analysis_arrays_metadata.json's temperature_K below
    centers = [0.0, 0.5]; ks = [10.0, 10.0]
    cv = np.array([0.0, 0.5])
    window = np.array([0, 1], dtype=int)
    v_pep = np.array([10.0, 12.0]); v_dih = np.array([5.0, 5.5])
    lam_sample = np.array([0.0, 1.0])   # window 0 -> lambda 0, window 1 -> lambda 1
    umbrella_kcal = np.array([
        0.5 * ks[0] * (cv[0] - centers[0]) ** 2,
        0.5 * ks[1] * (cv[1] - centers[1]) ** 2,
    ])
    u_umbrella_reduced = beta * 4.184 * np.vstack([
        np.array([umbrella_kcal[0], 0.5 * ks[1] * (cv[0] - centers[1]) ** 2]),
        np.array([0.5 * ks[0] * (cv[1] - centers[0]) ** 2, umbrella_kcal[1]]),
    ])
    np.savez(
        tmp_path / 'analysis_arrays.npz',
        cv_A=cv, window=window, replica=np.zeros(2, int), step=np.arange(2),
        umbrella_reduced_bias_nk=u_umbrella_reduced,
        v_pep_kj_mol=v_pep, v_dih_kj_mol=v_dih, gamd_lambda=lam_sample,
    )
    import csv as _csv
    with (tmp_path / 'umbrella_windows.csv').open('w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['center_A', 'k_kcal_mol_A2'])
        w.writeheader()
        for c, k in zip(centers, ks):
            w.writerow({'center_A': c, 'k_kcal_mol_A2': k})
    (tmp_path / 'analysis_arrays_metadata.json').write_text(json.dumps({'temperature_K': 300.0}))
    (tmp_path / 'shared_gamd_setup_globals.json').write_text(json.dumps({
        'all_globals': {'k0_Total': 0.8, 'Vmax_Total': 50.0, 'Vmin_Total': -50.0, 'threshold_energy_Total': 50.0,
                         'k0_Dihedral': 0.6, 'Vmax_Dihedral': 50.0, 'Vmin_Dihedral': -50.0, 'threshold_energy_Dihedral': 50.0}
    }))

    d = load_npz(tmp_path)
    assert d.meta.get('gamd_ladder') is True
    assert np.isfinite(d.u_nk).all()
    # column 0 (lambda=0) untouched; column 1 (lambda=1) carries the boost.
    np.testing.assert_allclose(d.u_nk[:, 0], u_umbrella_reduced[:, 0])
    expected_col1 = u_umbrella_reduced[:, 1] + beta * np.array([
        pep_gamd_boost_kj(v_pep[0], v_dih[0], 1.0, env),
        pep_gamd_boost_kj(v_pep[1], v_dih[1], 1.0, env),
    ])
    np.testing.assert_allclose(d.u_nk[:, 1], expected_col1)


def test_load_npz_leaves_u_untouched_when_ladder_arrays_absent(tmp_path):
    """No v_pep_kj_mol/v_dih_kj_mol/gamd_lambda in the npz (the pre-ladder,
    and current real-world, case) -> u must be exactly the stored array and
    meta['gamd_ladder'] must never be set."""
    from gareus.mbar_analysis.loaders import load_npz

    u_umbrella = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savez(
        tmp_path / 'analysis_arrays.npz',
        cv_A=np.array([0.0, 0.5]), window=np.array([0, 1], dtype=int),
        replica=np.zeros(2, int), step=np.arange(2),
        umbrella_reduced_bias_nk=u_umbrella,
    )
    import csv as _csv
    with (tmp_path / 'umbrella_windows.csv').open('w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['center_A', 'k_kcal_mol_A2'])
        w.writeheader()
        w.writerow({'center_A': 0.0, 'k_kcal_mol_A2': 10.0})
        w.writerow({'center_A': 0.5, 'k_kcal_mol_A2': 10.0})
    import json
    (tmp_path / 'analysis_arrays_metadata.json').write_text(json.dumps({'temperature_K': 300.0}))

    d = load_npz(tmp_path)
    np.testing.assert_array_equal(d.u_nk, u_umbrella)
    assert 'gamd_ladder' not in d.meta


# --- Task 7: lambda=0-only vs full-ladder PMF cross-check -------------------
# The lambda=0 rungs are plain umbrella sampling; their PMF, computed with the
# global f_k but only their own samples (subset N_k against the shared f_k --
# see _subset_logw_from_global_fk), must agree with the full-ladder PMF
# within error. This is the spec's quoting gate for whether the ladder
# boost's reweighting is right.

def test_ladder_crosscheck_agrees_when_lambda_zero_samples_are_representative():
    """Synthetic Data: identical (all-zero) bias in both states, so lambda=0
    and lambda=1 samples are draws from the same distribution -- the
    lambda=0-only PMF and the full-ladder PMF must coincide within noise."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.data import Data

    rng = np.random.default_rng(0)
    n = 4000
    cv = rng.normal(0.3, 0.05, n)
    window = np.repeat([0, 1], n // 2)
    u_nk = np.zeros((n, 2))
    beta = 1.0 / 2.494
    d = Data.__new__(Data)                     # minimal construction -- see data.py:92-117
    d.cv, d.window, d.u_nk, d.beta = cv, window, u_nk, beta
    d.state_lambdas = np.array([0.0, 1.0])
    d.meta = {"gamd_ladder": True}
    f_k = np.zeros(2)

    out = ladder_crosscheck(d, f_k, bins=20, kbt_kcal=0.596)
    assert out["status"] == "pass"
    assert out["max_abs_diff_kcal"] < 0.15
    assert out["n_lambda0_samples"] == n // 2
    assert out["tolerance_source"] == "fixed_default"


def test_ladder_crosscheck_skips_without_lambda_zero_states():
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.data import Data

    d = Data.__new__(Data)
    d.cv = np.zeros(10)
    d.window = np.zeros(10, int)
    d.u_nk = np.zeros((10, 1))
    d.beta = 0.4
    d.state_lambdas = np.array([1.0])
    d.meta = {"gamd_ladder": True}

    out = ladder_crosscheck(d, np.zeros(1), bins=5, kbt_kcal=0.596)
    assert out["status"] == "skipped"
    assert out["n_lambda0_samples"] == 0


def test_ladder_crosscheck_fails_when_the_tolerance_is_zero():
    """Exercises the 'fail' branch directly: the SAME representative data
    as the 'agrees' test above still has some nonzero counting noise
    (max_abs_diff_kcal > 0), so an artificially zero tolerance must flip
    the verdict to 'fail' while leaving every summary key populated."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.data import Data

    rng = np.random.default_rng(0)
    n = 4000
    cv = rng.normal(0.3, 0.05, n)
    window = np.repeat([0, 1], n // 2)
    u_nk = np.zeros((n, 2))
    beta = 1.0 / 2.494
    d = Data.__new__(Data)
    d.cv, d.window, d.u_nk, d.beta = cv, window, u_nk, beta
    d.state_lambdas = np.array([0.0, 1.0])
    d.meta = {"gamd_ladder": True}
    f_k = np.zeros(2)

    out = ladder_crosscheck(d, f_k, bins=20, kbt_kcal=0.596, tol_kcal=0.0)
    assert out["status"] == "fail"
    assert out["max_abs_diff_kcal"] > 0.0
    assert out["n_lambda0_samples"] == n // 2
    assert out["tolerance_kcal"] == 0.0
    assert out["n_bins_compared"] > 0


def test_ladder_crosscheck_failure_warning_is_triaged_critical():
    """gareus_report.py's _WARN_RULES must classify the exact warning text
    analyze_gareus_mbar.py emits on a failed cross-check as CRITICAL -- the
    binding constraint that a 'fail' surfaces as a report warning."""
    import gareus_report as gr

    groups = gr.classify_warnings([
        "λ-ladder cross-check FAILED: λ=0-only PMF disagrees with the full-ladder "
        "PMF by 1.234 kcal/mol (tolerance 0.500), over 2000 λ=0 samples -- the "
        "ladder boost reweighting embedded in u_nk does not reproduce plain "
        "umbrella sampling on its own rungs; every PMF from this run is suspect."])
    assert groups[0]["severity"] == "CRITICAL"


# --- Review fix round 1 (Critical 1 / Important 2): vacuous-pass guard and a
# discriminating (non-degenerate) test that can actually detect the forbidden
# naive masked-renormalize computation. ------------------------------------

def test_ladder_crosscheck_a_single_surviving_bin_is_not_a_vacuous_pass():
    """Critical-1 regression: with exactly ONE bin (bins=1), that bin IS the
    alignment reference (F_full[both].min() == F_full[that bin]), so the old
    code produced diff==0 identically -- 'pass' at ANY tolerance, including
    tol_kcal=0.0, with nothing but n_bins_compared==1 to betray it. The fix
    (MIN_BINS_FOR_VERDICT) must report 'skipped', never 'pass', when too few
    bins survive to make the comparison meaningful."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.data import Data

    rng = np.random.default_rng(0)
    n = 4000
    cv = rng.normal(0.3, 0.05, n)
    window = np.repeat([0, 1], n // 2)
    u_nk = np.zeros((n, 2))
    beta = 1.0 / 2.494
    d = Data.__new__(Data)
    d.cv, d.window, d.u_nk, d.beta = cv, window, u_nk, beta
    d.state_lambdas = np.array([0.0, 1.0])
    d.meta = {"gamd_ladder": True}
    f_k = np.zeros(2)

    out = ladder_crosscheck(d, f_k, bins=1, kbt_kcal=0.596, tol_kcal=0.0)
    assert out["status"] != "pass"
    assert out["status"] == "skipped"
    assert out["n_bins_compared"] == 1


def test_ladder_crosscheck_is_sensitive_to_the_forbidden_naive_reweight():
    """Important-2 regression: tests using an all-zero u_nk (degenerate --
    naive masked-renormalize and the correct _subset_logw_from_global_fk
    are algebraically identical there) cannot detect a regression that
    swapped the forbidden computation back into ladder_crosscheck. This one
    uses a REAL, non-degenerate multi-window harmonic-umbrella Data (the
    same _harmonic_windows_data builder validated in
    test_masked_logw_subset_pmf.py) with two λ=0 windows at different
    centres plus one λ>0 window elsewhere and a real MBAR-solved f_k.

    Asserts (a) the real ladder_crosscheck (using the correct subset
    reweight) passes with a small max_abs_diff_kcal, and (b) an
    independently-computed FORBIDDEN alternative -- masking the GLOBAL logw
    at the λ=0 rows and renormalizing, instead of recomputing the
    denominator with the subset's own N_k -- gives a materially (>10x)
    larger disagreement. This is the discriminator: if a future edit
    swapped the forbidden computation into ladder_crosscheck, (a) would
    fail because the reported max_abs_diff_kcal would jump to roughly the
    naive value asserted in (b).
    """
    from test_masked_logw_subset_pmf import _harmonic_windows_data
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    from gareus.mbar_analysis.solvers import solve_mbar, norm_logw
    from gareus.mbar_analysis.pmf import make_bins, pmf_from_weights
    import analyze_gareus_mbar as A

    rng = np.random.default_rng(7)
    centers = [0.0, 1.5, 3.0]
    k_spring = [40.0, 40.0, 40.0]
    beta = 0.4
    d = _harmonic_windows_data(rng, centers, k_spring, beta,
                               counts_per_window=[3000, 3000, 3000])
    d.state_lambdas = np.array([0.0, 0.0, 1.0])   # windows 0,1 lambda=0; window 2 the boosted rung
    d.meta["gamd_ladder"] = True

    m = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)
    assert m["converged"]
    bins = make_bins(d.cv, 30, None, None)
    kbt_kcal = (1.0 / beta) / A.KJ_PER_KCAL

    out = ladder_crosscheck(d, m["f_k"], bins, kbt_kcal)
    assert out["status"] == "pass"
    assert out["max_abs_diff_kcal"] < 0.1

    # Forbidden alternative, computed independently (not by monkeypatching
    # ladder_crosscheck): mask the GLOBAL logw at the lambda=0 rows and
    # renormalize -- the naive approach _subset_logw_from_global_fk exists
    # to replace.
    mask = np.isin(d.window, [0, 1])
    naive_w = norm_logw(m["logw"][mask])
    pmf_naive = pmf_from_weights(d.cv[mask], naive_w, bins, kbt_kcal)
    F_full = np.asarray(out["pmf_full"]["pmf"])
    F_naive = np.asarray(pmf_naive["pmf"])
    both = np.isfinite(F_full) & np.isfinite(F_naive)
    diff_naive = (F_full - F_full[both].min()) - (F_naive - F_naive[both].min())
    max_abs_naive = float(np.max(np.abs(diff_naive[both])))
    assert max_abs_naive > 1.0
    assert max_abs_naive > 10 * out["max_abs_diff_kcal"]

    # Pin the RETURNED pmf_lambda0 itself against the naive curve (not just
    # the reported max_abs_diff_kcal against pmf_full) -- a regression that
    # swapped the forbidden computation into ladder_crosscheck could in
    # principle change pmf_lambda0 while some other bookkeeping kept
    # max_abs_diff_kcal looking small; this closes that gap directly.
    F_lam0 = np.asarray(out["pmf_lambda0"]["pmf"])
    both_returned = np.isfinite(F_lam0) & np.isfinite(F_naive)
    with np.errstate(invalid="ignore"):
        diff_returned_vs_naive = ((F_lam0 - F_lam0[both_returned].min())
                                  - (F_naive - F_naive[both_returned].min()))
    assert float(np.max(np.abs(diff_returned_vs_naive[both_returned]))) > 1.0


# --- Final-review fix wave, C1: "gamd_ladder asserted but no λ in
# state_lambdas" is a contradiction, not a pass. -----------------------------

def _contradiction_data(state_lambdas):
    """Minimal Data whose meta claims an active ladder while state_lambdas
    carries no λ > 0 -- exactly what every loader that forgets to pass
    state_lambdas through produces (C2/C3)."""
    from gareus.mbar_analysis.data import Data

    rng = np.random.default_rng(0)
    n = 4000
    d = Data.__new__(Data)
    d.cv = rng.normal(0.3, 0.05, n)
    d.window = np.repeat([0, 1], n // 2)
    d.u_nk = np.zeros((n, 2))
    d.beta = 1.0 / 2.494
    d.state_lambdas = state_lambdas
    d.meta = {"gamd_ladder": True}
    return d


def test_ladder_crosscheck_missing_state_lambdas_is_not_a_vacuous_pass():
    """C1 regression. With meta['gamd_ladder'] True and state_lambdas None,
    the old code substituted np.zeros(K): EVERY column read λ=0, so the
    "λ=0-only subset" was the whole population, the two PMFs were
    bit-identical, and the gate returned status 'pass' with
    max_abs_diff_kcal == 0.0 -- a vacuous PASS that hid C2 and C3.

    Asserts on `status` (not on max_abs_diff_kcal, which reads a
    perfectly innocent 0.0 in the broken case), and that the status is one
    gareus_report._check_ladder_crosscheck grades FAIL -- 'skipped' would
    grade NA and would not block."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck
    import gareus_report as gr

    d = _contradiction_data(None)
    out = ladder_crosscheck(d, np.zeros(2), bins=20, kbt_kcal=0.596)
    assert out["status"] != "pass", out
    assert out["status"] == "fail", out
    assert "state_lambdas" in out.get("reason", "")
    graded = gr._check_ladder_crosscheck({"ladder_crosscheck": out})
    assert graded["status"] == gr.FAIL, graded


def test_ladder_crosscheck_all_zero_state_lambdas_under_active_ladder_fails():
    """Same contradiction reached the other way: state_lambdas present but
    all-zero. apply_ladder_boost_to_u only ever sets meta['gamd_ladder']
    True when some λ > 0, so an all-zero vector alongside that flag means a
    loader dropped the ladder on the floor."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck

    d = _contradiction_data(np.zeros(2))
    out = ladder_crosscheck(d, np.zeros(2), bins=20, kbt_kcal=0.596)
    assert out["status"] == "fail", out
    # No comparison was made, so the contradiction return deliberately carries
    # no max_abs_diff_kcal -- a 0.0 there is precisely the vacuous number the
    # broken code reported, and must not be reproduced by the fix.
    assert "max_abs_diff_kcal" not in out, out


def test_ladder_crosscheck_contradiction_guard_ignores_non_ladder_runs():
    """A plain umbrella run has meta['gamd_ladder'] False (or absent) and
    all-zero/absent state_lambdas -- that is not a contradiction, and must
    still reach the ordinary 'no λ=0 states'/comparison logic rather than
    the new FAIL."""
    from gareus.mbar_analysis.crosscheck import ladder_crosscheck

    d = _contradiction_data(None)
    d.meta = {"gamd_ladder": False}
    out = ladder_crosscheck(d, np.zeros(2), bins=20, kbt_kcal=0.596)
    assert out["status"] == "pass", out

    d2 = _contradiction_data(None)
    d2.meta = {}
    out2 = ladder_crosscheck(d2, np.zeros(2), bins=20, kbt_kcal=0.596)
    assert out2["status"] == "pass", out2


def test_ladder_crosscheck_contradiction_warning_text_is_triaged_critical():
    """C1, second half: analyze_gareus_mbar.py emits a DIFFERENT warning
    string for the contradiction 'fail' (it has no max_abs_diff_kcal to
    quote). That string must still be triaged CRITICAL by gareus_report's
    _WARN_RULES -- the rule keys on the "λ-ladder cross-check FAILED"
    prefix, which the contradiction text keeps."""
    import gareus_report as gr

    groups = gr.classify_warnings([
        "λ-ladder cross-check FAILED: meta['gamd_ladder'] is asserted but state_lambdas "
        "carries no λ > 0 (absent) -- the loader lost the per-state ladder rungs -- the "
        "cross-check could not be made at all, so no PMF from this run is certified."])
    assert groups[0]["severity"] == "CRITICAL"


# --- Final-review fix wave, C2: load_union_npz must carry state_lambdas /
# v_pep / v_dih through, and must map raw state ids to u_nk column indices.

def _write_union_npz(ap_dir, state_ids, state_lambdas, sampled_state_ids,
                     with_ladder_arrays=True):
    """Minimal adaptive_union_mbar.{npz,json} pair in the exact schema
    build_union_state_mbar_inputs writes (adaptive_production.py:3164-3201)."""
    import json

    ap_dir.mkdir(parents=True, exist_ok=True)
    state_ids = np.asarray(state_ids, dtype=np.int64)
    sampled_state_ids = np.asarray(sampled_state_ids, dtype=np.int64)
    K = state_ids.size
    n = sampled_state_ids.size
    id_to_index = {int(s): i for i, s in enumerate(state_ids.tolist())}
    n_k = np.zeros(K, dtype=np.int64)
    for sid in sampled_state_ids:
        n_k[id_to_index[int(sid)]] += 1
    rng = np.random.default_rng(3)
    cv = rng.normal(5.0, 0.5, n)
    arrays = dict(
        state_ids=state_ids,
        sampled_state_ids=sampled_state_ids,
        cv_A=cv,
        secondary_cv=np.full(n, np.nan),
        primary_centers=np.linspace(4.0, 6.0, K),
        primary_k=np.full(K, 10.0),
        secondary_centers=np.full(K, np.nan),
        secondary_k=np.zeros(K),
        umbrella_bias_kcal_mol_nk=np.zeros((n, K)),
        umbrella_bias_kj_mol_nk=np.zeros((n, K)),
        # Already carries the ladder boost: build_union_state_mbar_inputs runs
        # apply_ladder_boost_to_u BEFORE reducing by beta, so a loader that
        # added the term a second time would double-count it.
        umbrella_reduced_bias_nk=np.tile(np.arange(K, dtype=float), (n, 1)),
        N_k=n_k,
        gamd_boost_kj_nk=np.zeros((n, K)),
    )
    if with_ladder_arrays:
        arrays.update(
            state_lambdas=np.asarray(state_lambdas, dtype=np.float64),
            v_pep_kj_mol=np.full(n, 12.5),
            v_dih_kj_mol=np.full(n, 3.25),
        )
    np.savez_compressed(ap_dir / "adaptive_union_mbar.npz", **arrays)
    (ap_dir / "adaptive_union_mbar.json").write_text(json.dumps({
        "schema_version": "adaptive_union_mbar_inputs_v1",
        "beta_1_over_kJ_mol": 1.0 / 2.494,
        "state_ids": [int(x) for x in state_ids.tolist()],
        "N_k": [int(x) for x in n_k.tolist()],
        "gamd_ladder": bool(np.any(np.asarray(state_lambdas) > 0.0)),
    }))
    return n_k


def test_load_union_npz_maps_noncontiguous_state_ids_to_column_indices(tmp_path):
    """C2 regression. `window` is used directly as a u_nk COLUMN index and as
    the grouping key for every solver's np.bincount-derived N_k, but
    load_union_npz assigned it the RAW `sampled_state_ids`. Those coincide
    with column indices only when the ids are contiguous 0..K-1 -- state
    retirement leaves gaps, and then every sample is attributed to the wrong
    state (or out of range entirely)."""
    from gareus.mbar_analysis.loaders_adaptive import load_union_npz

    ap = tmp_path / "adaptive_production"
    state_ids = [0, 3, 7]                      # NON-contiguous: state 1,2,4.. retired
    lambdas = [0.0, 0.5, 1.0]
    sampled = [0, 0, 0, 3, 3, 7, 7, 7, 7, 0, 3, 7]
    n_k = _write_union_npz(ap, state_ids, lambdas, sampled)

    d = load_union_npz(ap)

    assert d.u_nk.shape[1] == 3
    assert d.window.max() < d.u_nk.shape[1], d.window
    # The discriminating assertion: the per-state counts the loader's own
    # `window` implies must equal the N_k the builder wrote next to the matrix.
    assert np.array_equal(np.bincount(d.window, minlength=3), n_k), (np.bincount(d.window, minlength=3), n_k)


def test_load_union_npz_carries_state_lambdas_and_raw_energies(tmp_path):
    """C2 regression, second half: without state_lambdas on Data the
    λ-ladder cross-check reads every column as λ=0 and passes vacuously
    (C1). v_pep/v_dih must arrive too -- they are what any downstream
    re-reweighting needs."""
    from gareus.mbar_analysis.loaders_adaptive import load_union_npz

    ap = tmp_path / "adaptive_production"
    _write_union_npz(ap, [0, 3, 7], [0.0, 0.5, 1.0], [0, 3, 7, 0, 3, 7])

    d = load_union_npz(ap)

    assert d.state_lambdas is not None
    assert np.allclose(d.state_lambdas, [0.0, 0.5, 1.0])
    assert d.v_pep_kj is not None and np.allclose(d.v_pep_kj, 12.5)
    assert d.v_dih_kj is not None and np.allclose(d.v_dih_kj, 3.25)
    assert d.meta.get("gamd_ladder") is True


def test_load_union_npz_does_not_re_add_the_boost(tmp_path):
    """The npz's umbrella_reduced_bias_nk ALREADY contains the ladder term
    (build_union_state_mbar_inputs calls apply_ladder_boost_to_u before
    reducing by beta). load_union_npz must pass state_lambdas through
    WITHOUT applying the boost a second time."""
    from gareus.mbar_analysis.loaders_adaptive import load_union_npz

    ap = tmp_path / "adaptive_production"
    _write_union_npz(ap, [0, 3, 7], [0.0, 0.5, 1.0], [0, 3, 7, 0, 3, 7])
    with np.load(ap / "adaptive_union_mbar.npz") as f:
        expected = np.asarray(f["umbrella_reduced_bias_nk"], dtype=float)

    d = load_union_npz(ap)
    assert np.allclose(d.u_nk, expected)


def test_load_union_npz_without_ladder_arrays_still_loads(tmp_path):
    """An npz written before the ladder columns existed has no
    state_lambdas/v_pep/v_dih -- it must still load, with all-zero lambdas
    and no ladder flag, rather than raising."""
    from gareus.mbar_analysis.loaders_adaptive import load_union_npz

    ap = tmp_path / "adaptive_production"
    _write_union_npz(ap, [0, 1, 2], [0.0, 0.0, 0.0], [0, 1, 2, 0, 1, 2],
                     with_ladder_arrays=False)

    d = load_union_npz(ap)
    assert d.u_nk.shape[1] == 3
    assert d.state_lambdas is None or not np.any(np.asarray(d.state_lambdas) > 0.0)
    assert not d.meta.get("gamd_ladder")


# --- Final-review fix wave, C3: _augment_with_adaptive_rounds rebuilds u_nk
# analytically (pure umbrella) and must re-apply the ladder boost afterwards.

def _write_windows_csv(d, rows):
    import csv as _csv
    d.mkdir(parents=True, exist_ok=True)
    fields = ["window", "center_A", "k_kcal_mol_A2", "primary_center", "primary_k",
              "secondary_cv_center", "secondary_cv_k_kcal_mol", "gamd_lambda"]
    with (d / "umbrella_windows.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _win_row(i, center, k, lam):
    return {"window": i, "center_A": center, "k_kcal_mol_A2": k,
            "primary_center": center, "primary_k": k,
            "secondary_cv_center": "", "secondary_cv_k_kcal_mol": "",
            "gamd_lambda": lam}


def _write_round_chunk(round_dir, cv, window, lam, v_pep, v_dih):
    chunks = round_dir / "analysis_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    n = len(cv)
    np.savez_compressed(
        chunks / "chunk_000000.npz",
        cv_A=np.asarray(cv, float),
        secondary_cv=np.full(n, np.nan),
        step=np.arange(n, dtype=np.int64),
        replica=np.zeros(n, dtype=np.int32),
        window=np.asarray(window, dtype=np.int32),
        gamd_boost_total_kj_mol=np.zeros(n),
        potential_kj_mol=np.full(n, np.nan),
        v_pep_kj_mol=np.asarray(v_pep, float),
        v_dih_kj_mol=np.asarray(v_dih, float),
        gamd_lambda=np.asarray(lam, float),
    )


def _ladder_run_with_rounds(tmp_path):
    """A run_dir with final_production/ + one adaptive_feedback_round_1/,
    both carrying a λ=0 and a λ=1 rung at the SAME umbrella centre."""
    from gareus.mbar_analysis.data import Data
    from gareus.pep_gamd import PepGamdEnvelope
    import json

    run = tmp_path / "run"
    prod = run / "final_production"
    rnd = run / "adaptive_feedback_round_1"
    rows = [_win_row(0, 5.0, 10.0, 0.0), _win_row(1, 5.0, 10.0, 1.0)]
    _write_windows_csv(prod, rows)
    _write_windows_csv(rnd, rows)

    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    run.mkdir(parents=True, exist_ok=True)
    (run / "shared_gamd_setup_globals.json").write_text(json.dumps({"all_globals": {
        "Vmax_Total": env.vmax_total, "Vmin_Total": env.vmin_total,
        "threshold_energy_Total": env.threshold_total, "k0_Total": env.k0max_total,
        "Vmax_Dihedral": env.vmax_dih, "Vmin_Dihedral": env.vmin_dih,
        "threshold_energy_Dihedral": env.threshold_dih, "k0_Dihedral": env.k0max_dih}}))

    _write_round_chunk(rnd, cv=[5.1, 5.2, 4.9, 5.05],
                       window=[0, 1, 0, 1], lam=[0.0, 1.0, 0.0, 1.0],
                       v_pep=[10.0, 11.0, 12.0, 13.0], v_dih=[3.0, 3.5, 4.0, 4.5])

    n = 4
    d = Data.__new__(Data)
    d.prod_dir = prod
    d.out_dir = prod / "pmf_analysis"
    d.cv = np.array([5.0, 5.15, 4.95, 5.2])
    d.cv2 = np.full(n, np.nan)
    d.rg_A = np.full(n, np.nan)
    d.window = np.array([0, 1, 0, 1])
    d.replica = np.zeros(n, int)
    d.step = np.arange(n)
    d.u_nk = np.zeros((n, 2))
    d.centers = np.array([5.0, 5.0])
    d.k_kcal = np.array([10.0, 10.0])
    d.beta = 1.0 / 2.494
    d.temp = 300.0
    d.boost_kj = np.zeros(n)
    d.potential_kj = np.full(n, np.nan)
    d.source = str(prod)
    d.meta = {"gamd_ladder": True}
    d.boost_dih_kj = None
    d.v_pep_kj = np.array([20.0, 21.0, 22.0, 23.0])
    d.v_dih_kj = np.array([5.0, 5.5, 6.0, 6.5])
    d.state_lambdas = np.array([0.0, 1.0])
    return run, d, env


def test_augment_with_rounds_keeps_lambda_rungs_as_distinct_union_states(tmp_path):
    """C3 regression. _build_union_window_table deduplicated on
    (primary_center, secondary_cv_center) only, so two rungs of the SAME
    window at different λ collapsed into one union state -- the ladder was
    erased from the state definition itself, not just from u_nk."""
    from gareus.mbar_analysis.loaders_adaptive import _augment_with_adaptive_rounds

    run, d, _env = _ladder_run_with_rounds(tmp_path)
    out = _augment_with_adaptive_rounds(d, run)

    assert out.u_nk.shape[1] == 2, out.u_nk.shape
    assert out.state_lambdas is not None
    assert sorted(np.asarray(out.state_lambdas).tolist()) == [0.0, 1.0]


def test_augment_with_rounds_re_applies_the_ladder_boost(tmp_path):
    """C3 regression, the real defect: u_all came from
    _compute_u_nk_analytical (pure umbrella, window dicts with no
    gamd_lambda), so the boost was discarded from every column while
    meta['gamd_ladder'] survived -- a wrong PMF AND (before C1) a vacuous
    cross-check PASS. The λ=1 column must now differ from the pure-umbrella
    reconstruction by exactly beta*pep_gamd_boost_kj per sample."""
    from gareus.mbar_analysis.loaders_adaptive import _augment_with_adaptive_rounds
    from gareus.pep_gamd import pep_gamd_boost_kj

    run, d, env = _ladder_run_with_rounds(tmp_path)
    out = _augment_with_adaptive_rounds(d, run)

    lam = np.asarray(out.state_lambdas)
    k1 = int(np.where(lam == 1.0)[0][0])
    k0 = int(np.where(lam == 0.0)[0][0])
    # Both union states share centre 5.0 / k 10.0, so their pure-umbrella
    # columns are identical -- the whole difference is the boost.
    expected = np.array([out.beta * pep_gamd_boost_kj(vp, vd, 1.0, env)
                         for vp, vd in zip(out.v_pep_kj, out.v_dih_kj)])
    assert np.any(expected > 0.0), expected
    assert np.allclose(out.u_nk[:, k1] - out.u_nk[:, k0], expected), (
        out.u_nk[:, k1] - out.u_nk[:, k0], expected)
    assert out.meta.get("gamd_ladder") is True


def test_augment_with_rounds_is_unchanged_for_a_non_ladder_run(tmp_path):
    """The guard must not perturb the ordinary (no-λ) multi-round path."""
    from gareus.mbar_analysis.loaders_adaptive import _augment_with_adaptive_rounds

    run, d, _env = _ladder_run_with_rounds(tmp_path)
    # Re-write both window tables with a single λ=0 window and clear the flag.
    for sub in ("final_production", "adaptive_feedback_round_1"):
        _write_windows_csv(run / sub, [_win_row(0, 5.0, 10.0, 0.0)])
    _write_round_chunk(run / "adaptive_feedback_round_1", cv=[5.1, 5.2],
                       window=[0, 0], lam=[0.0, 0.0],
                       v_pep=[np.nan, np.nan], v_dih=[np.nan, np.nan])
    d.meta = {}
    d.state_lambdas = np.array([0.0, 0.0])
    d.window = np.zeros(4, int)

    out = _augment_with_adaptive_rounds(d, run)
    assert out.u_nk.shape[1] == 1
    assert not out.meta.get("gamd_ladder")


# --- Final-review fix wave, I2/I1: the WRITER emits per-window gamd_lambda,
# and every loader prefers the written value over the nanmedian inference.

def test_snapshot_window_rows_carry_gamd_lambda():
    """I2 root cause: production.py's per-segment window snapshot emitted
    window_id/center1/k1/center2/k2 and no gamd_lambda, which is why three
    separate loaders had to INFER each state's rung by nanmedian over its
    samples' own gamd_lambda column."""
    from gareus.production import snapshot_window_rows

    rows = snapshot_window_rows([0.0, 1.0], [10.0, 20.0], None, None, [0.0, 1.0])
    assert [r["gamd_lambda"] for r in rows] == [0.0, 1.0]
    assert [r["window_id"] for r in rows] == [0, 1]
    assert "center2" not in rows[0]

    rows2d = snapshot_window_rows([0.0], [10.0], [2.0], [5.0], None)
    assert rows2d[0]["gamd_lambda"] == 0.0
    assert rows2d[0]["center2"] == 2.0 and rows2d[0]["k2"] == 5.0


def _parquet_ladder_run(prod, windows, lam_per_sample, with_envelope=True):
    """A minimal Parquet run: 2 windows sharing one umbrella centre, samples
    carrying real v_pep/v_dih, plus the frozen envelope next to it."""
    import json
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot
    from gareus.pep_gamd import PepGamdEnvelope

    prod.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(prod)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(prod).snapshot(seg_id, windows, cv1_type="distance", cv2_type=None)

    writer = ParquetSampleWriter(prod / "samples" / seg_id, flush_rows=1000)
    for i, lam in enumerate(lam_per_sample):
        writer.write_sample(i * 10, 0, i % len(windows), 5.0 + 0.01 * i, None,
                            -100.0, 1.0, 0.6, 0.4,
                            v_pep=10.0 + i, v_dih=3.0 + 0.5 * i, gamd_lambda=lam)
    writer.close()
    reg.close_segment(seg_id, end_step=10 * len(lam_per_sample))

    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    if with_envelope:
        (prod / "shared_gamd_setup_globals.json").write_text(json.dumps({"all_globals": {
            "Vmax_Total": env.vmax_total, "Vmin_Total": env.vmin_total,
            "threshold_energy_Total": env.threshold_total, "k0_Total": env.k0max_total,
            "Vmax_Dihedral": env.vmax_dih, "Vmin_Dihedral": env.vmin_dih,
            "threshold_energy_Dihedral": env.threshold_dih, "k0_Dihedral": env.k0max_dih}}))
    return env


def test_load_parquet_uses_the_written_gamd_lambda_and_does_not_raise(tmp_path):
    """I2: with gamd_lambda in the snapshot, load_parquet must (a) still
    load -- gareus.query.reconstruct_bias_matrix raises on a λ>0 window
    unless v_pep/v_dih/envelope are passed in, which is the trap adding the
    column opens -- and (b) take state_lambdas straight from the snapshot,
    with the boost applied exactly ONCE."""
    from gareus.mbar_analysis.loaders import load_parquet
    from gareus.pep_gamd import pep_gamd_boost_kj

    prod = tmp_path / "final_production"
    windows = [{"window_id": 0, "center1": 5.0, "k1": 10.0, "gamd_lambda": 0.0},
               {"window_id": 1, "center1": 5.0, "k1": 10.0, "gamd_lambda": 1.0}]
    env = _parquet_ladder_run(prod, windows, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    d = load_parquet(prod)

    assert np.allclose(d.state_lambdas, [0.0, 1.0])
    assert d.meta.get("gamd_ladder") is True
    assert d.meta.get("gamd_ladder_state_lambda_source") == "window_snapshot"
    expected = np.array([d.beta * pep_gamd_boost_kj(vp, vd, 1.0, env)
                         for vp, vd in zip(d.v_pep_kj, d.v_dih_kj)])
    assert np.any(expected > 0.0), expected
    # Both windows share centre/k, so their umbrella columns are identical and
    # the whole column difference is exactly one boost -- not two.
    assert np.allclose(d.u_nk[:, 1] - d.u_nk[:, 0], expected), (d.u_nk[:, 1] - d.u_nk[:, 0], expected)


def test_load_parquet_falls_back_to_nanmedian_for_pre_column_snapshots(tmp_path):
    """I2: a snapshot written before gamd_lambda existed carries no such key;
    the per-sample nanmedian inference must still recover each window's rung,
    and must record that it did so in meta."""
    from gareus.mbar_analysis.loaders import load_parquet

    prod = tmp_path / "final_production"
    windows = [{"window_id": 0, "center1": 5.0, "k1": 10.0},
               {"window_id": 1, "center1": 5.0, "k1": 10.0}]      # no gamd_lambda key
    _parquet_ladder_run(prod, windows, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    d = load_parquet(prod)

    assert np.allclose(d.state_lambdas, [0.0, 1.0])
    assert d.meta.get("gamd_ladder") is True
    assert d.meta.get("gamd_ladder_state_lambda_source") == "per_sample_nanmedian_fallback"


def test_load_parquet_non_ladder_run_is_untouched(tmp_path):
    from gareus.mbar_analysis.loaders import load_parquet

    prod = tmp_path / "final_production"
    windows = [{"window_id": 0, "center1": 5.0, "k1": 10.0, "gamd_lambda": 0.0},
               {"window_id": 1, "center1": 5.5, "k1": 10.0, "gamd_lambda": 0.0}]
    _parquet_ladder_run(prod, windows, [0.0] * 6, with_envelope=False)

    d = load_parquet(prod)
    assert not np.any(np.asarray(d.state_lambdas) > 0.0)
    assert d.meta.get("gamd_ladder") is False


def test_load_epoch_csv_adaptive_keys_states_on_gamd_lambda(tmp_path):
    """I1: _row_state_key was (center1, k1, center2, k2) with no
    gamd_lambda, so every rung of one window merged into ONE state and λ was
    then a nanmedian over a mixture of rungs."""
    import csv as _csv
    import json
    from gareus.mbar_analysis.loaders_adaptive import load_epoch_csv_adaptive
    from gareus.pep_gamd import PepGamdEnvelope

    root = tmp_path / "run"
    ap = root / "adaptive_production"
    epoch = ap / "epoch_000" / "baseline"
    epoch.mkdir(parents=True, exist_ok=True)
    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    (ap / "shared_gamd_setup_globals.json").write_text(json.dumps({"all_globals": {
        "Vmax_Total": env.vmax_total, "Vmin_Total": env.vmin_total,
        "threshold_energy_Total": env.threshold_total, "k0_Total": env.k0max_total,
        "Vmax_Dihedral": env.vmax_dih, "Vmin_Dihedral": env.vmin_dih,
        "threshold_energy_Dihedral": env.threshold_dih, "k0_Dihedral": env.k0max_dih}}))

    # secondary_cv_center is written as a real 0.0 rather than left blank: on a
    # blank, _row_state_key's _fkey falls back to a FRESH float('nan') per call,
    # and a NaN key element is not equal to itself, so the second pass raises
    # KeyError. That is a pre-existing defect of this (dormant, legacy-CSV-only)
    # loader, unrelated to I1 and deliberately not fixed in this wave.
    fields = ["cv_A", "secondary_cv", "secondary_cv_center", "step", "replica", "window",
              "center_A", "k_kcal_mol_A2", "beta_1_over_kJ_mol", "gamd_boost_total_kj_mol",
              "potential_kj_mol", "v_pep_kj_mol", "v_dih_kj_mol", "gamd_lambda"]
    with (epoch / "samples.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(8):
            lam = 0.0 if i % 2 == 0 else 1.0
            w.writerow({"cv_A": 5.0 + 0.01 * i, "secondary_cv": "",
                        "secondary_cv_center": 0.0, "step": i * 10,
                        "replica": 0, "window": i % 2, "center_A": 5.0,
                        "k_kcal_mol_A2": 10.0, "beta_1_over_kJ_mol": 1.0 / 2.494,
                        "gamd_boost_total_kj_mol": 0.0, "potential_kj_mol": -100.0,
                        "v_pep_kj_mol": 10.0 + i, "v_dih_kj_mol": 3.0 + 0.5 * i,
                        "gamd_lambda": lam})

    d = load_epoch_csv_adaptive(ap)

    # Same centre and k for both rungs: without λ in the key this is ONE state.
    assert d.u_nk.shape[1] == 2, d.u_nk.shape
    assert sorted(np.asarray(d.state_lambdas).tolist()) == [0.0, 1.0]
    assert d.meta.get("gamd_ladder") is True


def test_export_analysis_arrays_npz_survives_a_ladder_snapshot(tmp_path):
    """I2 follow-on: export_analysis_arrays_npz is the SECOND caller that
    reconstructs from the window snapshot. Once snapshot_window_rows writes
    gamd_lambda, it hits the same query.py:395-400 ValueError load_parquet
    did -- and it is a public entry point (see gareus/helptext.py), not only
    an internal one. It must pass the raw energies through and emit the same
    total bias MBAR consumes, not an umbrella-only matrix."""
    from gareus.query import export_analysis_arrays_npz
    from gareus.pep_gamd import pep_gamd_boost_kj

    prod = tmp_path / "final_production"
    windows = [{"window_id": 0, "center1": 5.0, "k1": 10.0, "gamd_lambda": 0.0},
               {"window_id": 1, "center1": 5.0, "k1": 10.0, "gamd_lambda": 1.0}]
    env = _parquet_ladder_run(prod, windows, [0.0, 1.0, 0.0, 1.0])

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(prod, beta)
    with np.load(out, allow_pickle=False) as f:
        nk = np.asarray(f["umbrella_reduced_bias_nk"], dtype=float)
        v_pep = np.asarray(f["v_pep_kj_mol"], dtype=float) if "v_pep_kj_mol" in f.files else None

    assert nk.shape[1] == 2
    if v_pep is None:
        v_pep = np.array([10.0 + i for i in range(nk.shape[0])])
    v_dih = np.array([3.0 + 0.5 * i for i in range(nk.shape[0])])
    expected = np.array([beta * pep_gamd_boost_kj(vp, vd, 1.0, env)
                         for vp, vd in zip(v_pep, v_dih)])
    assert np.any(expected > 0.0)
    assert np.allclose(nk[:, 1] - nk[:, 0], expected), (nk[:, 1] - nk[:, 0], expected)

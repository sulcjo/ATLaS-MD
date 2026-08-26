"""Mapping-sanity diagnostics added after the chignolin_6 root-cause hunt.

Every test here is a regression guard for something that produced a full,
plausible-looking PMF on a real multi-day run while being catastrophically
wrong, with not one existing check firing. The reference numbers are lifted
from docs/chignolin_6_low_ess_root_cause.md; see it for the full evidence
chain.

Covers, in the order the fixes were requested:

* A4  -- per-state self-bias (own samples scored in their own restraint).
         The single check that would have caught the whole root cause.
* A12 -- window_diagnostics.csv exposing the SECONDARY restraint + observed
         cv2, i.e. the axis that actually failed on that run.
* A6  -- worst-overlap pairs reported by CV-space adjacency, not by state
         index (the reported "pair 23-24" sent the investigation at an
         innocent, well-overlapped window).
* A5  -- the overlap matrix itself being joint (cv1, cv2) rather than the
         cv1 marginal, which could not see the gap that had opened.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.mbar_analysis.pmf as pmfmod  # noqa: E402
import gareus_report as gr  # noqa: E402


# ===========================================================================
# fixtures
# ===========================================================================
class _Args:
    bins = 20
    min_neighbor_overlap = 0.30
    selected_method = 'auto'
    gamd_smooth_sigma = 0.0
    pmf_smooth_sigma = 0.0
    smooth_sigma = 0.0
    pmf_uncertainty = False
    pmf_uncertainty_n_boot = 10
    pmf_uncertainty_seed = 0
    cv2_bins = None
    cv2_min = None
    cv2_max = None


def _harmonic_u_nk(cv, cv2, centers, k_kcal, centers2, k2, beta_kcal):
    """Reduced (kT) bias matrix for a primary + optional secondary harmonic.

    Mirrors every loader's own construction (0.5*k*(x-c)^2 summed over the
    restrained axes, then scaled by beta) so u_nk here is in the same units
    the real pipeline stores -- dimensionless kT.
    """
    n = cv.size
    K = len(centers)
    u = np.zeros((n, K), dtype=np.float64)
    for k in range(K):
        tot = 0.5 * k_kcal[k] * (cv - centers[k]) ** 2
        if centers2 is not None and np.isfinite(centers2[k]) and k2[k] > 0.0:
            tot = tot + 0.5 * k2[k] * (cv2 - centers2[k]) ** 2
        u[:, k] = beta_kcal * tot
    return u


def _make_data(cv, cv2, window, centers, k_kcal, centers2=None, k2=None,
               out_dir=None, kbt_kcal=0.6):
    """A Data whose u_nk is a real harmonic bias matrix in kT."""
    from gareus.mbar_analysis.data import Data
    from gareus.units import KJ_PER_KCAL
    beta_kcal = 1.0 / kbt_kcal                     # 1/kT, per kcal/mol
    beta = beta_kcal / KJ_PER_KCAL                 # 1/kT, per kJ/mol
    n = cv.size
    meta = {}
    if centers2 is not None:
        meta['umbrella_window_rows'] = [
            {'center_A': str(centers[k]), 'k_kcal_mol_A2': str(k_kcal[k]),
             'secondary_center': str(centers2[k]), 'secondary_k': str(k2[k])}
            for k in range(len(centers))
        ]
    u = _harmonic_u_nk(cv, cv2, centers, k_kcal, centers2, k2, beta_kcal)
    return Data(
        prod_dir=Path(out_dir or '.'), out_dir=Path(out_dir or '.'),
        cv=cv, cv2=cv2, rg_A=np.full(n, np.nan), window=window,
        replica=window.astype(np.int64), step=np.arange(n, dtype=np.int64),
        u_nk=u, centers=np.asarray(centers, float), k_kcal=np.asarray(k_kcal, float),
        beta=beta, temp=1.0 / (beta * 0.008314462618), boost_kj=np.full(n, np.nan),
        potential_kj=None, source='test', meta=meta,
    )


def _clean_2d_run(n_per=4000, seed=0, kbt_kcal=0.6):
    """3 states on a well-resolved 2D ladder, each sampling its OWN restraint.

    spacing / sigma is ~1 on both axes, i.e. the textbook-healthy geometry
    the real run did NOT have.
    """
    rng = np.random.default_rng(seed)
    k_p, k_s = 100.0, 100.0
    s_p = np.sqrt(kbt_kcal / k_p)
    s_s = np.sqrt(kbt_kcal / k_s)
    centers = np.array([0.0, s_p, 2.0 * s_p])
    centers2 = np.array([0.0, s_s, 2.0 * s_s])
    k_kcal = np.full(3, k_p)
    k2 = np.full(3, k_s)
    cv, cv2, win = [], [], []
    for k in range(3):
        cv.append(rng.normal(centers[k], s_p, n_per))
        cv2.append(rng.normal(centers2[k], s_s, n_per))
        win.append(np.full(n_per, k, dtype=np.int64))
    return (np.concatenate(cv), np.concatenate(cv2), np.concatenate(win),
            centers, k_kcal, centers2, k2)


# ===========================================================================
# A4 -- self-bias sanity check
# ===========================================================================
def test_self_bias_of_a_correctly_mapped_2dof_harmonic_is_about_one_kT():
    """u_nk is already reduced, so a state's own samples in its own 2-DOF
    restraint follow Exp(1) in kT: median ln2 = 0.693, mean 1, p90 2.303.

    This is the anchor for the whole check -- it pins both the units (kT, no
    further beta) and the "~1 kT" claim the thresholds are hung off.
    """
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=20000, seed=1)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2)
    sb = pmfmod.self_bias_diagnostics(d.u_nk, d.window, 3)
    med = np.asarray(sb['median_kT'], float)
    p90 = np.asarray(sb['p90_kT'], float)
    assert np.all(np.abs(med - np.log(2.0)) < 0.05), med
    assert np.all(np.abs(p90 - 2.303) < 0.15), p90
    assert list(np.asarray(sb['n'], int)) == [20000] * 3
    assert pmfmod.self_bias_warning_lines(sb) == []


def test_self_bias_catches_a_state_scored_against_the_wrong_centre():
    """chignolin_6's state 21: its slot was fed real-state-22 samples, so it
    was scored against a sign-flipped secondary centre (median 63.6 kT).
    """
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=3000, seed=2)
    # Same shape as the real bug: state 1's secondary centre is flipped, so its
    # own samples now sit many sigma away from the restraint they are scored in.
    centers2_broken = centers2.copy()
    centers2_broken[1] = -centers2[1] - 1.0
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2_broken, k2)
    sb = pmfmod.self_bias_diagnostics(d.u_nk, d.window, 3)
    med = np.asarray(sb['median_kT'], float)
    assert med[1] > pmfmod.SELF_BIAS_MEDIAN_WARN_KT
    assert med[0] < 2.0 and med[2] < 2.0
    lines = pmfmod.self_bias_warning_lines(sb)
    assert len(lines) == 1
    assert 'state 1' in lines[0]
    assert 'chignolin_6_low_ess_root_cause.md' in lines[0]
    # must land in the operator's HIGH bucket, not the unmatched MEDIUM default
    assert gr._severity_of(lines[0]) == 'HIGH'


def test_self_bias_catches_a_pooled_hamiltonian_via_p90_not_median():
    """chignolin_6's state 22: median 1.1 kT (textbook) but p90 652.8, because
    its slot pooled three different Hamiltonians and only the minority is
    fabricated. A median-only check misses this state entirely.
    """
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=1000, seed=3)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2)
    # Contaminate 15% of state 1's rows with samples from far up the cv2 axis.
    rows = np.flatnonzero(d.window == 1)
    bad = rows[: int(0.15 * rows.size)]
    d.cv2[bad] = centers2[1] + 40.0 * np.sqrt(0.6 / k2[1])
    d.u_nk = _harmonic_u_nk(d.cv, d.cv2, centers, k_kcal, centers2, k2, 1.0 / 0.6)
    sb = pmfmod.self_bias_diagnostics(d.u_nk, d.window, 3)
    med = np.asarray(sb['median_kT'], float)
    p90 = np.asarray(sb['p90_kT'], float)
    assert med[1] < pmfmod.SELF_BIAS_MEDIAN_WARN_KT      # median alone looks fine
    assert p90[1] > pmfmod.SELF_BIAS_P90_WARN_KT         # p90 is what catches it
    lines = pmfmod.self_bias_warning_lines(sb)
    assert len(lines) == 1 and 'state 1' in lines[0]


def test_self_bias_is_nan_and_silent_for_a_state_with_no_samples():
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=500, seed=4)
    keep = win != 2
    d = _make_data(cv[keep], cv2[keep], win[keep], centers, k_kcal, centers2, k2)
    sb = pmfmod.self_bias_diagnostics(d.u_nk, d.window, 3)
    assert int(np.asarray(sb['n'], int)[2]) == 0
    assert not np.isfinite(np.asarray(sb['median_kT'], float)[2])
    assert pmfmod.self_bias_warning_lines(sb) == []


def test_self_bias_ignores_nonfinite_bias_rows():
    """A NaN u_nk entry (an excluded sample) must not poison a state's median
    into NaN and thereby silence the check for that state."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=400, seed=5)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2)
    d.u_nk[5, 0] = np.nan
    sb = pmfmod.self_bias_diagnostics(d.u_nk, d.window, 3)
    assert int(np.asarray(sb['n'], int)[0]) == 399
    assert np.isfinite(np.asarray(sb['median_kT'], float)[0])


def test_self_bias_warning_and_columns_reach_the_end_to_end_report(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=800, seed=6)
    centers2_broken = centers2.copy()
    centers2_broken[1] = -centers2[1] - 1.0
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2_broken, k2, out_dir=tmp_path)
    warns: list = []
    info = pmfmod.run_pmf_and_gamd_boost_report(
        d, _Args(), np.zeros(cv.size), np.linspace(-0.1, 0.3, 21), 0.6,
        tmp_path, warns, None)
    assert any('Implausible self-bias' in w for w in warns), warns
    assert 'self_bias' in info
    with (tmp_path / 'window_diagnostics.csv').open() as f:
        rows = {int(r['window']): r for r in csv.DictReader(f)}
    assert float(rows[1]['self_bias_median_kT']) > pmfmod.SELF_BIAS_MEDIAN_WARN_KT
    assert float(rows[0]['self_bias_median_kT']) < 2.0
    assert rows[0]['self_bias_p90_kT'] != ''


# ===========================================================================
# A12 -- secondary restraint visible in window_diagnostics.csv
# ===========================================================================
def test_window_diagnostics_exposes_secondary_centre_k_and_observed_cv2(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=900, seed=7)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    pmfmod.run_pmf_and_gamd_boost_report(
        d, _Args(), np.zeros(cv.size), np.linspace(-0.1, 0.3, 21), 0.6,
        tmp_path, [], None)
    with (tmp_path / 'window_diagnostics.csv').open() as f:
        rdr = csv.DictReader(f)
        assert 'secondary_center_window_table' in rdr.fieldnames
        assert 'secondary_k_kcal_mol_window_table' in rdr.fieldnames
        assert 'cv2_mean' in rdr.fieldnames
        assert 'cv2_std' in rdr.fieldnames
        rows = {int(r['window']): r for r in rdr}
    for k in range(3):
        assert float(rows[k]['secondary_center_window_table']) == pytest.approx(centers2[k])
        assert float(rows[k]['secondary_k_kcal_mol_window_table']) == pytest.approx(k2[k])
        assert float(rows[k]['cv2_mean']) == pytest.approx(centers2[k], abs=0.02)
        assert float(rows[k]['cv2_std']) == pytest.approx(np.sqrt(0.6 / k2[k]), rel=0.15)


def test_window_diagnostics_secondary_columns_are_blank_for_a_cv1_only_run(tmp_path):
    rng = np.random.default_rng(8)
    centers = np.array([0.0, 0.5, 1.0]); k_kcal = np.full(3, 10.0)
    cv = np.concatenate([rng.normal(c, 0.25, 700) for c in centers])
    win = np.concatenate([np.full(700, k, dtype=np.int64) for k in range(3)])
    d = _make_data(cv, np.full(cv.size, np.nan), win, centers, k_kcal, out_dir=tmp_path)
    pmfmod.run_pmf_and_gamd_boost_report(
        d, _Args(), np.zeros(cv.size), np.linspace(-1.0, 2.0, 21), 0.6,
        tmp_path, [], None)
    with (tmp_path / 'window_diagnostics.csv').open() as f:
        rows = {int(r['window']): r for r in csv.DictReader(f)}
    for k in range(3):
        assert rows[k]['secondary_center_window_table'] == ''
        assert rows[k]['secondary_k_kcal_mol_window_table'] == ''
        assert rows[k]['cv2_mean'] == '' and rows[k]['cv2_std'] == ''


def test_the_secondary_columns_are_named_for_the_snapshot_they_come_from(tmp_path):
    """M1: those two columns are ONE window-table snapshot; their neighbours are not.

    ``_secondary_window_params`` reads ``meta['umbrella_window_rows']`` -- for
    the union loader that is ``final_registry_used_for_mbar.csv``, i.e. a single
    snapshot applied to every epoch. ``cv2_mean`` beside it is observed and
    ``self_bias_median_kT`` is reconstructed from each epoch's OWN native
    params, so on a run that recentred its windows mid-campaign (the tICA CV2
    switch overwrites centres in place) the snapshot and the observed value
    legitimately disagree -- in a table whose stated new purpose is mapping
    sanity, where an unexplained disagreement reads as a mapping fault.

    The header now says which is which, so this asserts the column named for
    the snapshot really carries the snapshot rather than the observed value.
    """
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=900, seed=11)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    # Recentre exactly the way the CV2 switch does: the window rows the analysis
    # is handed no longer describe where these samples were collected.
    recentred = np.asarray(centers2) + 0.75
    for k, row in enumerate(d.meta['umbrella_window_rows']):
        row['secondary_center'] = str(recentred[k])

    pmfmod.run_pmf_and_gamd_boost_report(
        d, _Args(), np.zeros(cv.size), np.linspace(-0.1, 0.3, 21), 0.6,
        tmp_path, [], None)

    with (tmp_path / 'window_diagnostics.csv').open() as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames)
        rows = {int(r['window']): r for r in rdr}
    # The provenance-free spelling is gone, not merely duplicated -- a reader
    # cannot land on the ambiguous name any more.
    assert 'secondary_center' not in fields
    assert 'secondary_k_kcal_mol' not in fields
    for k in range(3):
        snapshot = float(rows[k]['secondary_center_window_table'])
        observed = float(rows[k]['cv2_mean'])
        assert snapshot == pytest.approx(recentred[k])
        assert observed == pytest.approx(centers2[k], abs=0.02)
        # Fixture check: the two really do disagree here, so the header is
        # carrying real information rather than restating the same number.
        assert abs(snapshot - observed) > 0.5


def test_secondary_window_params_reads_every_loader_spelling():
    """The three loaders that populate meta['umbrella_window_rows'] use three
    different column spellings for the same quantity; all must resolve."""
    meta = {'umbrella_window_rows': [
        {'secondary_center': '1.5', 'secondary_k': '200'},
        {'secondary_cv_center': '2.5', 'secondary_cv_k_kcal_mol': '150'},
        {'center_A': '0.0', 'k_kcal_mol_A2': '10'},          # CV1-only row
    ]}
    c2, k2 = pmfmod._secondary_window_params(meta, 3)
    assert c2[0] == pytest.approx(1.5) and k2[0] == pytest.approx(200.0)
    assert c2[1] == pytest.approx(2.5) and k2[1] == pytest.approx(150.0)
    assert not np.isfinite(c2[2]) and not np.isfinite(k2[2])


def test_secondary_window_params_pads_a_short_or_absent_row_list():
    c2, k2 = pmfmod._secondary_window_params({}, 4)
    assert c2.size == 4 and not np.any(np.isfinite(c2))
    c2, k2 = pmfmod._secondary_window_params(
        {'umbrella_window_rows': [{'secondary_center': '1.0', 'secondary_k': '5'}]}, 3)
    assert c2.size == 3 and np.isfinite(c2[0]) and not np.isfinite(c2[2])


# ===========================================================================
# A6 -- CV-space adjacency, not index adjacency
# ===========================================================================
def test_cv_space_neighbour_reduces_to_index_adjacency_on_a_1d_ladder():
    """A monotone 1D ladder's nearest neighbour in centre space IS its index
    neighbour, so this fix must be a no-op there."""
    K = 5
    centers = np.linspace(0.0, 1.0, K); k_kcal = np.full(K, 20.0)
    O = np.eye(K) + 0.4 * (np.abs(np.subtract.outer(np.arange(K), np.arange(K))) == 1)
    rows = pmfmod.cv_space_neighbor_overlap(
        O, centers, k_kcal, np.full(K, np.nan), np.full(K, np.nan), 0.6,
        n_k=np.full(K, 100))
    got = {r['window']: r['neighbor'] for r in rows}
    assert got == {0: 1, 1: 0, 2: 1, 3: 2, 4: 3}   # ties resolve to lowest index
    assert all(r['overlap'] == pytest.approx(0.4) for r in rows)


def test_cv_space_neighbour_ignores_index_adjacency_when_cv2_separates():
    """The chignolin_6 misdirection, reproduced: state 3 sits next to state 4 in
    INDEX order but is far from it in (cv1, cv2) space, while its true nearest
    neighbour (state 0) is well separated too. The index-adjacent pair 3-4 must
    not be the one reported.
    """
    centers = np.array([0.00, 0.00, 0.00, 0.20, 0.02])
    centers2 = np.array([0.00, 0.10, 0.20, 3.00, 0.04])
    k_kcal = np.full(5, 100.0)
    k2 = np.full(5, 100.0)
    O = np.full((5, 5), 0.9); np.fill_diagonal(O, 1.0)
    O[3, 4] = O[4, 3] = 0.013          # the innocent-looking index-adjacent pair
    O[0, 3] = O[3, 0] = 0.55           # state 3's real nearest neighbour in CV space
    rows = pmfmod.cv_space_neighbor_overlap(
        O, centers, k_kcal, centers2, k2, 0.6, n_k=np.full(5, 100))
    got = {r['window']: r['neighbor'] for r in rows}
    # 4 is nearest to 0 (both axes close); 3 is nowhere near 4 on the cv2 axis
    assert got[4] == 0
    assert got[3] != 4
    worst = min(r['overlap'] for r in rows)
    assert worst > 0.013, 'the index-adjacent 3-4 pair must not set the headline'


def test_cv_space_neighbour_scales_each_axis_by_its_own_sigma():
    """Raw Euclidean distance would be dominated by whichever axis happens to
    have the larger numeric range; the metric must be in units of the states'
    own restraint widths (sigma ~ sqrt(kT/k)) instead."""
    # cv1 centres 0.10 apart with a very stiff restraint (sigma 0.0245) are
    # 4 sigma apart; cv2 centres 1.0 apart with a very soft one (sigma 0.775)
    # are only 1.3 sigma apart. Raw Euclidean would call cv2 the bigger gap.
    centers = np.array([0.0, 0.10, 0.0])
    centers2 = np.array([0.0, 0.0, 1.0])
    k_kcal = np.array([1000.0, 1000.0, 1000.0])
    k2 = np.array([1.0, 1.0, 1.0])
    O = np.full((3, 3), 0.5); np.fill_diagonal(O, 1.0)
    rows = pmfmod.cv_space_neighbor_overlap(
        O, centers, k_kcal, centers2, k2, 0.6, n_k=np.full(3, 100))
    by_w = {r['window']: r for r in rows}
    assert by_w[0]['neighbor'] == 2          # 1.3 sigma away, not the 4-sigma one
    assert by_w[0]['centre_distance_sigma'] < by_w[1]['centre_distance_sigma']


def test_cv_space_neighbour_skips_states_with_no_samples():
    """A zero-sample state has an all-zero overlap row; letting it be somebody's
    reported neighbour would make the headline 0.000 for a reason the dedicated
    zero-window check already reports."""
    K = 3
    centers = np.array([0.0, 0.1, 0.2]); k_kcal = np.full(K, 20.0)
    O = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.0], [0.2, 0.0, 0.0]])
    rows = pmfmod.cv_space_neighbor_overlap(
        O, centers, k_kcal, np.full(K, np.nan), np.full(K, np.nan), 0.6,
        n_k=np.array([100, 100, 0]))
    assert {r['window'] for r in rows} == {0, 1}
    assert all(r['neighbor'] != 2 for r in rows)


def test_cv_space_neighbour_overlap_reaches_the_end_to_end_report(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=900, seed=9)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    info = pmfmod.run_pmf_and_gamd_boost_report(
        d, _Args(), np.zeros(cv.size), np.linspace(-0.1, 0.3, 21), 0.6,
        tmp_path, [], None)
    rows = info['cv_space_neighbor_overlap']
    assert isinstance(rows, list) and len(rows) == 3
    assert all({'window', 'neighbor', 'overlap'} <= set(r) for r in rows)
    assert isinstance(info['neighbor_overlap'], list)   # index array still there
    with (tmp_path / 'window_diagnostics.csv').open() as f:
        rdr = csv.DictReader(f)
        assert 'cv_space_neighbor' in rdr.fieldnames
        assert 'cv_space_overlap_marginal' in rdr.fieldnames


# ===========================================================================
# A6 -- health verdict consumes the CV-space number
# ===========================================================================
def _summary(**over):
    s = {
        'n_samples': 1_000_000, 'n_windows': 10,
        'selected_unbiased_method': 'umbrella_only',
        'mbar': {'converged': True, 'max_delta': 1e-9, 'n_k': [100_000] * 10,
                 'base_ess': 300_000.0},
        'neighbor_overlap': [0.5] * 9,
        'boost': {'available': False},
        'warnings': [],
    }
    s.update(over)
    return s


def test_health_verdict_prefers_cv_space_overlap_over_index_adjacency():
    s = _summary(cv_space_neighbor_overlap=[
        {'window': 0, 'neighbor': 1, 'overlap': 0.04},
        {'window': 1, 'neighbor': 0, 'overlap': 0.04},
    ])
    s['neighbor_overlap'] = [0.9] * 9          # index view says everything is fine
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'fail'
    assert ov['metric'] == pytest.approx(0.04)
    assert 'CV-space' in ov['detail']
    assert '0-1' in ov['detail']


def test_health_verdict_index_fallback_says_the_pair_is_index_adjacent():
    """Without the CV-space list the old number is all there is -- but the
    detail string must no longer imply the pair is physically adjacent, which
    is exactly what misdirected the chignolin_6 investigation."""
    s = _summary()
    s['neighbor_overlap'] = [0.5, 0.5, 0.013] + [0.5] * 6
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'fail'
    assert 'index-adjacent' in ov['detail']
    assert '0.013' in ov['detail']


def test_health_verdict_cv_space_overlap_passing_is_pass():
    s = _summary(cv_space_neighbor_overlap=[
        {'window': 0, 'neighbor': 1, 'overlap': 0.62},
        {'window': 1, 'neighbor': 0, 'overlap': 0.62},
    ])
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'pass'


def test_health_verdict_ignores_a_malformed_cv_space_list():
    """Never raise on a partial/old-format summary (the module's own contract)."""
    for junk in ([], [{'window': 0}], 'nonsense', [{'overlap': None}]):
        s = _summary(cv_space_neighbor_overlap=junk)
        v = gr.build_health_verdict(s, 0.30)
        ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
        assert ov['status'] in ('pass', 'caution', 'fail', 'na')


def test_self_bias_warning_is_triaged_high():
    groups = gr.classify_warnings([
        'Implausible self-bias in 3 umbrella state(s): state 21 (median 63.6 kT, '
        'p90 80.7 kT). See docs/chignolin_6_low_ess_root_cause.md.'])
    assert groups[0]['severity'] == 'HIGH'


def test_weak_cv_space_overlap_warning_is_triaged_high():
    groups = gr.classify_warnings([
        'Weak CV-space nearest-neighbour overlap below 0.30 (2D histogram) for '
        'pairs: 20-21 (0.05)'])
    assert groups[0]['severity'] == 'HIGH'


# ===========================================================================
# A5 / FINDING 2 -- the joint (CV1, CV2) overlap is published ALONGSIDE the
# marginal, never silently in place of it.
#
# Round 1 made the joint overlap the reported number under the SAME key
# (s['neighbor_overlap']), the same window_diagnostics.csv columns and the same
# overlap_matrix.csv/.png file names the marginal had always used, gated by the
# SAME --min-neighbor-overlap threshold (0.30, calibrated against marginal
# numbers), with no space marker anywhere on disk and no opt-out. Consequences
# that these tests exist to prevent coming back:
#
#   * two runs analysed either side of that change were not comparable, and
#     nothing written to disk said which space a number was in;
#   * joint overlap is bounded above by the CV1 marginal (min of a refinement
#     sums to no more than min of the coarsening), so every already-published
#     2D run would start emitting new "Weak neighbor CV overlap below 0.30"
#     warnings -- triaged HIGH by gareus_report -- and flip its top-level
#     PASS/CAUTION/FAIL verdict on data that had not changed.
#
# The shape pinned below: the marginal keeps every historical key, column, file
# and threshold; the joint is published beside it under its own names, with its
# own dimension-aware threshold and a space stamp on disk; the health verdict
# consumes the joint knowingly; and there is a flag to turn it off.
# ===========================================================================
def _cv1_degenerate_2d_run(n_per=3000, seed=10, kbt_kcal=0.6):
    """chignolin_6's real geometry in miniature: every state shares ONE primary
    centre and they differ only in CV2, ~13 sigma apart.

    So the CV1 marginal overlap is ~1.0 (reads perfectly healthy, which is
    exactly what the real run reported) while the joint overlap is ~0.
    """
    rng = np.random.default_rng(seed)
    K = 3
    centers = np.zeros(K)
    k_kcal = np.full(K, 100.0)
    centers2 = np.array([0.0, 1.0, 2.0])
    k2 = np.full(K, 100.0)
    s_p = np.sqrt(kbt_kcal / 100.0); s_s = np.sqrt(kbt_kcal / 100.0)
    cv = np.concatenate([rng.normal(0.0, s_p, n_per) for _ in range(K)])
    cv2 = np.concatenate([rng.normal(c, s_s, n_per) for c in centers2])
    win = np.concatenate([np.full(n_per, k, dtype=np.int64) for k in range(K)])
    return cv, cv2, win, centers, k_kcal, centers2, k2


def _report(d, tmp_path, warns=None, bins=None, **argover):
    args = _Args()
    for k, v in argover.items():
        setattr(args, k, v)
    return pmfmod.run_pmf_and_gamd_boost_report(
        d, args, np.zeros(d.cv.size),
        np.linspace(-0.3, 0.3, 21) if bins is None else bins, 0.6,
        tmp_path, [] if warns is None else warns, None)


def test_the_reported_overlap_matrix_stays_the_cv1_marginal_for_a_2d_run(tmp_path):
    """Continuity: `O`, `neighbor_overlap` and overlap_matrix.csv must hold the
    same numbers a pre-2026-08-25 analysis of the same run produced, bit for
    bit -- not the joint numbers."""
    from gareus.mbar_analysis.solvers import overlap_matrix
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=30)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    bins = np.linspace(-0.3, 0.3, 21)
    info = _report(d, tmp_path, bins=bins)
    ref = overlap_matrix(d.cv, d.window, bins, 3)          # 4-arg = pre-fix path
    assert np.array_equal(info['O'], ref)
    assert info['neighbor_overlap'] == [float(ref[i, i + 1]) for i in range(2)]
    assert info['overlap_space'] == 'cv1_marginal'


def test_a_2d_run_gains_no_new_weak_marginal_overlap_warning(tmp_path):
    """The reviewer's concrete failure: re-analysing an already-published 2D run
    must not start emitting the pre-existing (HIGH-triaged) marginal warning
    just because the joint space is now also measured. The joint weakness is
    reported, under its own distinct wording and its own threshold."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=31)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    warns: list = []
    info = _report(d, tmp_path, warns=warns)
    # CV1-marginal overlap of these states is ~1.0, so the historical warning
    # must stay silent (it is computed on the marginal, as it always was).
    assert not any('Weak neighbor CV overlap' in w for w in warns), warns
    assert min(info['neighbor_overlap']) > 0.30
    # ... while the real, joint-space gap IS reported, distinctly worded so
    # gareus_report can triage it separately from the marginal warning.
    joint_warns = [w for w in warns if 'joint' in w.lower() and 'overlap' in w.lower()]
    assert joint_warns, warns
    assert not any('Weak neighbor CV overlap' in w for w in joint_warns)


def test_joint_overlap_is_published_alongside_under_its_own_key(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=32)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    info = _report(d, tmp_path)
    jo = info['joint_overlap']
    assert jo['available'] is True
    assert jo['space'] == 'cv1_cv2_joint'
    assert jo['dim'] == 2
    # The joint matrix sees the gap the marginal cannot.
    assert max(jo['neighbor_overlap']) < 0.05
    assert min(info['neighbor_overlap']) > 0.9
    rows = {int(r['window']): r for r in jo['cv_space_neighbor_overlap']}
    assert set(rows) == {0, 1, 2}
    assert all(r['overlap'] < 0.05 for r in rows.values())
    assert jo['secondary_cv_finite_fraction'] == pytest.approx(1.0)
    assert (tmp_path / 'overlap_matrix_joint.csv').exists()
    assert info['files']['overlap_matrix_joint_csv'] == str(tmp_path / 'overlap_matrix_joint.csv')


def test_both_overlap_csvs_stamp_which_space_their_numbers_are_in(tmp_path):
    """"There is no way to tell them apart" was half the finding: a number's
    space must be self-describing on disk, not inferable only from the commit
    the analysis ran at."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=33)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    _report(d, tmp_path)
    marg = (tmp_path / 'overlap_matrix.csv').read_text().splitlines()
    assert marg[0].startswith('#')
    assert 'CV1 marginal' in marg[0]
    assert marg[1].startswith('window,')          # header row itself unchanged
    joint = (tmp_path / 'overlap_matrix_joint.csv').read_text().splitlines()
    assert joint[0].startswith('#')
    assert 'joint (CV1, CV2)' in joint[0]
    assert joint[1].startswith('window,')


def test_the_joint_threshold_is_its_own_number_not_min_neighbor_overlap(tmp_path):
    """--min-neighbor-overlap was calibrated against marginal numbers. The
    default joint target is that same per-axis quality applied to both axes
    (thr**dim), and it is separately overridable."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=34)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    info = _report(d, tmp_path)
    assert info['joint_overlap']['threshold'] == pytest.approx(0.30 ** 2)
    assert info['joint_overlap']['threshold'] != 0.30
    info2 = _report(d, tmp_path, min_joint_neighbor_overlap=0.5)
    assert info2['joint_overlap']['threshold'] == pytest.approx(0.5)


def test_joint_overlap_can_be_turned_off(tmp_path):
    """An opt-out that restores the exact pre-2026-08-25 behaviour."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=35)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    warns: list = []
    info = _report(d, tmp_path, warns=warns, no_joint_overlap=True)
    assert info['joint_overlap']['available'] is False
    assert 'disabled' in info['joint_overlap']['reason'].lower()
    assert not (tmp_path / 'overlap_matrix_joint.csv').exists()
    assert 'overlap_matrix_joint_csv' not in info['files']
    assert not any('Weak joint' in w for w in warns), warns


def test_window_diagnostics_keeps_marginal_columns_and_names_the_joint_ones(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(seed=36)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    info = _report(d, tmp_path)
    O = info['O']
    with (tmp_path / 'window_diagnostics.csv').open() as f:
        rdr = csv.DictReader(f)
        for col in ('overlap_left', 'overlap_right',
                    'cv_space_neighbor', 'cv_space_overlap_marginal',
                    'cv_space_overlap_joint'):
            assert col in rdr.fieldnames, rdr.fieldnames
        rows = {int(r['window']): r for r in rdr}
    # overlap_left/right stay the MARGINAL numbers they have always been.
    assert float(rows[1]['overlap_left']) == pytest.approx(float(O[0, 1]))
    assert float(rows[1]['overlap_right']) == pytest.approx(float(O[1, 2]))
    # marginal ~1.0, joint ~0 for the same pair: the two columns are not the
    # same quantity and the header now says so.
    assert float(rows[1]['cv_space_overlap_marginal']) > 0.9
    assert float(rows[1]['cv_space_overlap_joint']) < 0.05


def test_joint_overlap_is_unavailable_with_a_reason_for_a_cv1_only_run(tmp_path):
    """d.cv2 is a NaN-filled array, never None, so the 2D decision cannot be a
    None check (CLAUDE.md, 2026-08-15). An all-NaN cv2 must leave the reported
    matrix on the exact marginal path and skip the joint entirely -- silently,
    since a 1D run has no secondary axis to be blind to."""
    rng = np.random.default_rng(11)
    centers = np.array([0.0, 0.4, 0.8]); k_kcal = np.full(3, 20.0)
    cv = np.concatenate([rng.normal(c, 0.2, 900) for c in centers])
    win = np.concatenate([np.full(900, k, dtype=np.int64) for k in range(3)])
    d = _make_data(cv, np.full(cv.size, np.nan), win, centers, k_kcal, out_dir=tmp_path)
    warns: list = []
    bins = np.linspace(-0.6, 1.4, 21)
    info = _report(d, tmp_path, warns=warns, bins=bins)
    assert info['joint_overlap']['available'] is False
    assert 'secondary' in info['joint_overlap']['reason'].lower()
    from gareus.mbar_analysis.solvers import overlap_matrix
    assert np.array_equal(info['O'], overlap_matrix(d.cv, d.window, bins, 3))
    # Silently: a 1D run has no secondary axis to be blind to, so it must not
    # be told about a joint matrix it could never have had. (This fixture's own
    # marginal overlap is genuinely weak, so marginal warnings do fire.)
    assert not any('joint' in w.lower() for w in warns), warns


def test_joint_overlap_is_skipped_when_windows_have_cv2_but_no_secondary_k(tmp_path):
    """Real cv2 values with no secondary restraint anywhere means cv2 is an
    observable, not a biased axis -- overlapping in it would report a gap the
    sampler was never asked to bridge."""
    rng = np.random.default_rng(12)
    centers = np.array([0.0, 0.4, 0.8]); k_kcal = np.full(3, 20.0)
    cv = np.concatenate([rng.normal(c, 0.2, 900) for c in centers])
    cv2 = rng.normal(0.0, 3.0, cv.size)
    win = np.concatenate([np.full(900, k, dtype=np.int64) for k in range(3)])
    d = _make_data(cv, cv2, win, centers, k_kcal,
                   centers2=np.full(3, np.nan), k2=np.zeros(3), out_dir=tmp_path)
    info = _report(d, tmp_path, bins=np.linspace(-0.6, 1.4, 21))
    assert info['joint_overlap']['available'] is False


def test_joint_overlap_is_skipped_when_most_samples_lack_a_secondary_cv(tmp_path):
    """A mid-campaign CV2 regime switch can legitimately null a large block of
    cv2. Below half the population the joint matrix describes a minority while
    looking just as authoritative, so it is not computed -- and, because this
    run DOES restrain CV2, the resulting blind spot is stated out loud."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=1200, seed=13)
    cv2 = cv2.copy()
    cv2[: int(0.7 * cv2.size)] = np.nan
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    warns: list = []
    info = _report(d, tmp_path, warns=warns, bins=np.linspace(-0.1, 0.3, 21))
    assert info['joint_overlap']['available'] is False
    assert '30' in info['joint_overlap']['reason']      # 30% retained
    assert any('NOT computed' in w for w in warns), warns


def test_joint_overlap_is_computed_but_flagged_for_a_small_cv2_gap(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=1200, seed=14)
    cv2 = cv2.copy()
    cv2[: int(0.10 * cv2.size)] = np.nan     # 90% retained: joint, but flagged
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    warns: list = []
    info = _report(d, tmp_path, warns=warns, bins=np.linspace(-0.1, 0.3, 21))
    assert info['joint_overlap']['available'] is True
    assert info['joint_overlap']['secondary_cv_finite_fraction'] == pytest.approx(0.9, abs=0.01)
    assert any('secondary' in w.lower() for w in warns), warns

# ===========================================================================
# FINDING 3 -- the joint-overlap memory guard must be sized in BYTES
#
# Round 1 sized it in CELLS (OVERLAP_CHUNK_CELLS = 8192), which never engages
# at the window counts this repo has real datasets for: in joint space
# n_cells = B * B2 = 60 * 30 = 1800 < 8192, so `step` became 1800 and the whole
# K x K x n_cells transient was materialised in one block regardless of K. For
# the 2D-rough oracle geometry (K = 364) that is
# 364^2 * 1800 * 8 B = 1.91 GB for a single np.minimum() broadcast, against
# 64 MB for the pre-fix marginal path -- default-on for any 2D run, in a module
# whose peak memory was already the subject of a dedicated audit. K = 27
# (chignolin_6) is only 10 MB, which is why local test data hid it.
# ===========================================================================
def test_chunk_step_bound_folds_k_in_and_bounds_the_real_oracle_geometry():
    """K=364 (the 2D-rough oracle) x 60x30 joint cells: the transient the guard
    permits must be a small multiple of the pre-fix marginal path's 64 MB, not
    the 1.91 GB an unchunked reduction would take."""
    from gareus.mbar_analysis import solvers as sv
    K, n_cells = 364, 60 * 30
    unchunked_bytes = K * K * n_cells * 8
    assert unchunked_bytes > 1.9e9                       # the regression, in bytes
    step = sv._overlap_chunk_cells(K, n_cells)
    assert 1 <= step <= n_cells
    permitted = K * K * step * 8
    assert permitted <= sv.OVERLAP_CHUNK_BYTES
    assert permitted < unchunked_bytes / 8               # real, not nominal, bounding


def test_chunk_step_never_drops_below_one_cell_at_absurd_k():
    """A K so large that even one cell exceeds the budget must still make
    progress (one cell per chunk) rather than divide down to a 0 step."""
    from gareus.mbar_analysis import solvers as sv
    assert sv._overlap_chunk_cells(100_000, 1800) == 1
    assert sv._overlap_chunk_cells(0, 1800) >= 1


def test_chunk_step_does_not_split_a_small_k_problem():
    """Sizing in bytes must not make the K=27 case (chignolin_6) slower than it
    was: 27^2 * 1800 * 8 = 10.5 MB fits the budget, so it stays one chunk."""
    from gareus.mbar_analysis import solvers as sv
    assert sv._overlap_chunk_cells(27, 60 * 30) >= 60 * 30


def test_the_joint_reduction_actually_honours_the_byte_budget(monkeypatch):
    """Instrumented: every array np.minimum() materialises inside the joint
    reduction must be within the budget. Proves the loop honours the bound
    rather than just that a helper computes one."""
    from gareus.mbar_analysis import solvers as sv
    K, B, B2 = 364, 20, 10
    budget = 8 * 1024 * 1024
    seen: list = []
    real_minimum = np.minimum

    def spy(a, b, *rest, **kw):
        r = real_minimum(a, b, *rest, **kw)
        seen.append(int(np.asarray(r).nbytes))
        return r

    monkeypatch.setattr(sv.np, 'minimum', spy)
    rng = np.random.default_rng(21)
    n = 20_000
    cv = rng.uniform(0.0, 1.0, n)
    cv2 = rng.uniform(-1.0, 1.0, n)
    window = rng.integers(0, K, n)
    sv.overlap_matrix(cv, window, np.linspace(0.0, 1.0, B + 1), K,
                      cv2=cv2, bins2=np.linspace(-1.0, 1.0, B2 + 1),
                      max_chunk_bytes=budget)
    assert seen, 'the joint reduction never called np.minimum'
    assert max(seen) <= budget, f'peak transient {max(seen)} B exceeds budget {budget} B'
    assert max(seen) < K * K * B * B2 * 8      # i.e. it really did chunk


# ===========================================================================
# FINDING 2 (cont.) -- the health verdict must consume the joint number
# KNOWINGLY: with its own threshold, saying which space it is in, and never
# letting a healthy-looking marginal hide a broken joint (or vice versa).
# ===========================================================================
def test_health_verdict_uses_the_joint_number_and_its_own_threshold():
    """The chignolin_6 illusion in one assertion: marginal 0.97 (passes 0.30),
    joint 0.02 (fails 0.09). The verdict must fail, and its detail must name
    both numbers so the operator can see why the marginal looked fine."""
    s = _summary(
        cv_space_neighbor_overlap=[{'window': 0, 'neighbor': 1, 'overlap': 0.97},
                                  {'window': 1, 'neighbor': 0, 'overlap': 0.97}],
        joint_overlap={'available': True, 'space': 'cv1_cv2_joint', 'dim': 2,
                       'threshold': 0.09,
                       'cv_space_neighbor_overlap': [
                           {'window': 0, 'neighbor': 1, 'overlap': 0.02},
                           {'window': 1, 'neighbor': 0, 'overlap': 0.02}]})
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'fail'
    assert ov['metric'] == pytest.approx(0.02)
    assert 'joint' in ov['detail'].lower()
    assert '0.020' in ov['detail'] and '0.97' in ov['detail']
    assert '0.09' in ov['detail']


def test_health_verdict_takes_the_worse_of_the_two_spaces():
    """Joint overlap is bounded above by the marginal, so a passing joint does
    not imply a passing marginal (joint 0.20 > 0.09 while marginal 0.25 < 0.30).
    Adding the joint check must not weaken the marginal one."""
    s = _summary(
        cv_space_neighbor_overlap=[{'window': 0, 'neighbor': 1, 'overlap': 0.25}],
        joint_overlap={'available': True, 'threshold': 0.09, 'dim': 2,
                       'cv_space_neighbor_overlap': [
                           {'window': 0, 'neighbor': 1, 'overlap': 0.20}]})
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'caution'
    # Both numbers must be visible: a detail quoting only one of them means one
    # of the two spaces was dropped rather than combined.
    assert 'joint' in ov['detail'].lower()
    assert '0.200' in ov['detail'] and '0.250' in ov['detail']


def test_health_verdict_falls_back_when_the_joint_is_unavailable():
    s = _summary(
        cv_space_neighbor_overlap=[{'window': 0, 'neighbor': 1, 'overlap': 0.62}],
        joint_overlap={'available': False, 'reason': 'no secondary restraint'})
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'pass'
    assert 'joint' not in ov['detail'].lower()


def test_weak_joint_overlap_warning_is_triaged_high_and_distinctly():
    joint = ('Weak joint (CV1, CV2) nearest-neighbour overlap below 0.09 for pairs: '
             '20-21 (0.02)')
    groups = gr.classify_warnings([joint])
    assert groups[0]['severity'] == 'HIGH'
    # It must NOT be collapsed into, or matched by, the pre-existing marginal
    # rule -- that is what would have inflated old runs' warning counts.
    assert gr._severity_of('Weak neighbor CV overlap below 0.30 for pairs: 1-2 (0.1)') == 'HIGH'
    both = gr.classify_warnings([joint, 'Weak neighbor CV overlap below 0.30 for pairs: 1-2 (0.1)'])
    assert len(both) == 2


def test_uncomputed_joint_overlap_is_high_but_an_opt_out_is_only_info():
    """A 2D run whose joint overlap could not be computed is genuinely blind on
    the axis that failed on chignolin_6 -- worth surfacing. A user who asked for
    that with --no-joint-overlap is not a defect."""
    assert gr._severity_of(
        'Joint (CV1, CV2) window overlap was NOT computed: only 30.0% of samples '
        'carry a finite secondary CV.') == 'HIGH'
    assert gr._severity_of(
        'Joint (CV1, CV2) window overlap is disabled by --no-joint-overlap; the '
        'reported overlap is the CV1 marginal.') == 'INFO'


# ===========================================================================
# FINDING 4 -- the stale-window-map escape hatch must not be able to sit inside
# a PASS verdict, and the mid-campaign CV2 regime change must be triaged on
# purpose rather than defaulting to MEDIUM.
#
# Wording below is copied from the real emitters
# (gareus/mbar_analysis/loaders_adaptive.py, loaders_union_parquet.py) so these
# rules are pinned against the strings the pipeline actually produces.
# ===========================================================================
_STALE_OVERRIDE_NOTE = (
    "[stale window map] epoch_window_map.csv for adaptive-production phase "
    "final/baseline lists 27 window(s) but the phase physically ran 24 (the phase's "
    "own recorded window count). No repair source reconciles. Loaded with the STALE "
    "map anyway because GAREUS_ALLOW_STALE_WINDOW_MAP is set: this phase's samples "
    "are attributed to the wrong umbrella states and every free energy derived from "
    "them is invalid. See docs/chignolin_6_low_ess_root_cause.md.")

_STALE_REPAIRED_NOTE = (
    "[stale window map] epoch_window_map.csv for adaptive-production phase "
    "final/baseline lists 27 window(s) but the phase physically ran 24 -- "
    "`--us-auto-drop-bad-windows` dropped windows post-pull and renumbered the "
    "survivors 0..23, but this map was never rewritten. REPAIRED IN MEMORY from the "
    "phase's own surviving window table; corrected local->state mapping for the 4 "
    "shifted window(s). The run data on disk is unchanged and still stale.")

_CV2_REGIME_NOTE = (
    "[cv2 regime change] The secondary CV was redefined mid-campaign: "
    "'torsion-pca' (epoch_000; 3,131,000 samples) then 'tica-linear' (epoch_001; "
    "5,050,000 samples), per each phase's own run_manifest.json. ... state k is NOT "
    "one Hamiltonian across the switch")


def test_the_stale_map_escape_hatch_is_critical():
    groups = gr.classify_warnings([_STALE_OVERRIDE_NOTE])
    assert groups[0]['severity'] == 'CRITICAL'


def test_the_stale_map_escape_hatch_cannot_sit_inside_a_pass_verdict():
    """The operator forgets the variable is exported; the next campaign's
    pmf_summary.json must not be able to read PASS while its samples are
    knowingly attributed to the wrong umbrella states."""
    s = _summary(warnings=[_STALE_OVERRIDE_NOTE])
    v = gr.build_health_verdict(s, 0.30)
    mp = next(c for c in v['checks'] if c['name'] == 'Sample-to-state mapping')
    assert mp['status'] == 'fail'
    assert 'GAREUS_ALLOW_STALE_WINDOW_MAP' in mp['detail']
    assert v['overall'] == 'FAIL'


def test_an_in_memory_repaired_stale_map_is_high_but_not_a_failed_verdict():
    """A repair means THIS analysis is correct (the on-disk map is still stale,
    which is why it is loud) -- so it must not be graded like the escape hatch."""
    assert gr.classify_warnings([_STALE_REPAIRED_NOTE])[0]['severity'] == 'HIGH'
    s = _summary(warnings=[_STALE_REPAIRED_NOTE])
    mp = next(c for c in gr.build_health_verdict(s, 0.30)['checks']
              if c['name'] == 'Sample-to-state mapping')
    assert mp['status'] == 'caution'


def test_the_cv2_regime_change_note_is_triaged_deliberately():
    """Not MEDIUM-by-default: a pooled MBAR solve across a secondary-CV
    redefinition is formally invalid across the boundary, per the note's own
    text."""
    assert gr.classify_warnings([_CV2_REGIME_NOTE])[0]['severity'] == 'HIGH'


def test_the_mapping_check_fails_on_implausible_self_bias_from_json():
    """Fed the ROUND-TRIPPED shape (wjson turns NaN into null and arrays into
    lists), because that is what gareus_report sees in production."""
    from gareus.mbar_analysis.data import wjson
    import json as _json
    sb = pmfmod.self_bias_diagnostics(
        np.array([[0.7, 40.0], [0.9, 41.0], [40.0, 63.6]]),
        np.array([0, 0, 1], dtype=np.int64), 2)
    p = Path(__import__('tempfile').mkdtemp()) / 'pmf_summary.json'
    wjson(p, _summary(self_bias=sb))
    s = _json.loads(p.read_text())
    assert s['self_bias']['median_kT'][1] == pytest.approx(63.6)
    mp = next(c for c in gr.build_health_verdict(s, 0.30)['checks']
              if c['name'] == 'Sample-to-state mapping')
    assert mp['status'] == 'fail'
    assert '63.6' in mp['detail'] or 'state 1' in mp['detail']


def test_the_mapping_check_is_na_when_nothing_is_known():
    mp = next(c for c in gr.build_health_verdict(_summary(), 0.30)['checks']
              if c['name'] == 'Sample-to-state mapping')
    assert mp['status'] == 'na'


def test_the_mapping_check_passes_on_healthy_self_bias():
    sb = pmfmod.self_bias_diagnostics(
        np.array([[0.7, 40.0], [0.9, 41.0], [40.0, 1.1]]),
        np.array([0, 0, 1], dtype=np.int64), 2)
    mp = next(c for c in gr.build_health_verdict(_summary(self_bias=sb), 0.30)['checks']
              if c['name'] == 'Sample-to-state mapping')
    assert mp['status'] == 'pass'


# ===========================================================================
# FINDING 1 -- the whole chain, through the REAL analyze().
#
# Round 1 added cv_space_neighbor_overlap / self_bias / the joint overlap to
# run_pmf_and_gamd_boost_report's return value and never edited analyze()'s
# s={...} summary literal, so NONE of it reached pmf_summary.json:
# build_health_verdict got a dict without those keys, _cv_space_worst returned
# None, and the "Window overlap" row silently fell through to the very
# index-adjacency branch the fix existed to replace. Every round-1 test went
# green anyway because they all fed hand-built dicts carrying a key the
# pipeline never emitted. Verified against this branch before the fix:
# `analyze()` really did report
# "worst 0.425 (windows 1-2, index-adjacent ...)".
#
# So this test drives the real analyze() -> pmf_summary.json -> gareus_report
# chain and reads the numbers back off disk. It is the only test here that can
# fail if a future edit drops the propagation again.
# ===========================================================================
def test_analyze_publishes_every_mapping_diagnostic_end_to_end(tmp_path):
    import json
    import analyze_gareus_mbar as agm

    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=1200, seed=40)
    # One state scored against a badly wrong secondary centre: the chignolin_6
    # failure in miniature (there, a sign-flipped centre2 with k2=148.5 produced
    # 2,012 kT of fabricated self-bias).
    centers2_broken = centers2.copy()
    centers2_broken[1] = -centers2[1] - 1.0
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2_broken, k2, out_dir=tmp_path)
    # Force the epoch_000/rest split so the epoch_000_separate report is built
    # too -- its summary block must carry the same diagnostics as the main one.
    # Interleaved, not sliced in half: the fixture concatenates state by state,
    # so a contiguous split would put whole states on one side and the main
    # report would see fewer than all three.
    src = (np.arange(cv.size) % 2).astype(np.int64)
    d.meta['_epoch_source'] = src.tolist()
    d.meta['adaptive_epoch_run_dirs'] = [str(tmp_path / 'epoch_000'), str(tmp_path / 'epoch_001')]

    s = agm.analyze(d, agm.parse_args([str(tmp_path)]), None)
    disk = json.loads((tmp_path / 'pmf_summary.json').read_text())

    # --- the diagnostics reached pmf_summary.json at all -------------------
    for key in ('overlap_space', 'neighbor_overlap', 'cv_space_neighbor_overlap',
                'self_bias', 'joint_overlap', 'secondary_cv_finite_fraction',
                'overlap_connectivity'):
        assert key in disk, f'{key} missing from pmf_summary.json'
    assert disk['overlap_space'] == 'cv1_marginal'
    assert len(disk['cv_space_neighbor_overlap']) == 3
    assert disk['joint_overlap']['available'] is True
    assert disk['self_bias']['median_kT'][1] > pmfmod.SELF_BIAS_MEDIAN_WARN_KT
    # Both spaces' connectivity survives the trip through JSON (this run is a
    # healthy 1-sigma ladder, so both are connected -- the split case is
    # covered by the unit and report-level tests above).
    conn = disk['overlap_connectivity']
    assert conn['marginal']['available'] is True and conn['marginal']['n_components'] == 1
    assert conn['joint']['available'] is True and conn['joint']['n_components'] == 1

    # --- the health verdict actually used them ----------------------------
    checks = {c['name']: c for c in disk['health']['checks']}
    ov = checks['Window overlap']
    # What this assertion is for (round 2): the pair NAMED as physically
    # adjacent must be the CV-space one -- presenting an index-order pair as
    # adjacent is what misdirected the original investigation. Round 3 folded
    # the index-adjacent number back in as an additional GRADED number (a
    # passing CV-space pair must not be able to hide a broken index ladder), so
    # it appears in `detail` too -- explicitly labelled as possibly-not-
    # adjacent, which is what is checked here now.
    assert 'CV-space nearest' in ov['detail'], ov['detail']
    assert 'index-adjacent (windows' in ov['detail'], ov['detail']
    assert 'may not be neighbours in CV space' in ov['detail'], ov['detail']
    assert 'joint' in ov['detail'].lower(), ov['detail']
    # M4: `metric` alone cannot say which space it was measured in, and the two
    # carry different thresholds. The label has to survive the trip onto disk --
    # a health block assembled field-by-field somewhere would drop it silently.
    assert ov['metric_space'] == pmfmod.OVERLAP_SPACE_JOINT, ov
    assert ov['metric'] == pytest.approx(
        min(r['overlap'] for r in disk['joint_overlap']['cv_space_neighbor_overlap']))
    assert checks['Overlap connectivity']['status'] == 'pass', checks['Overlap connectivity']
    mp = checks['Sample-to-state mapping']
    assert mp['status'] == 'fail', mp
    assert disk['health']['overall'] == 'FAIL'

    # --- the self-bias warning made it through triage ---------------------
    assert any('Implausible self-bias' in w for w in disk['warnings'])
    sev = {g['severity'] for g in disk['warnings_grouped']
           if 'Implausible self-bias' in g['representative']}
    assert sev == {'HIGH'}

    # --- the epoch_000 block carries the same fields (single source) -------
    e0 = disk['epoch_000_report']
    assert e0['available'] is True
    for key in ('overlap_space', 'neighbor_overlap', 'cv_space_neighbor_overlap',
                'self_bias', 'joint_overlap', 'secondary_cv_finite_fraction',
                'overlap_connectivity',
                'selected_unbiased_method', 'pmf_span_kcal_mol', 'boost'):
        assert key in e0, f'{key} missing from epoch_000_report'
    assert set(agm._report_summary_fields(_report(d, tmp_path / 'probe'))) <= set(disk)

    # --- and the on-disk matrices say which space they are in -------------
    assert 'CV1 marginal' in (tmp_path / 'overlap_matrix.csv').read_text().splitlines()[0]
    assert (tmp_path / 'overlap_matrix_joint.csv').exists()
    assert 'overlap_matrix_joint_csv' in disk['files']
    assert 'epoch_000_overlap_matrix_joint_csv' in disk['files']
    assert s['health']['overall'] == disk['health']['overall']


def test_the_joint_overlap_cli_flags_reach_the_report(tmp_path):
    """Argparse dest names vs the getattr names pmf.py reads -- checked against
    a REAL parse_args() object, not a stand-in _Args. A flag whose dest does not
    match what the consumer reads is silently inert, which is the same class of
    defect as round 1's unpropagated summary keys."""
    import analyze_gareus_mbar as agm
    defaults = agm.parse_args([str(tmp_path)])
    assert defaults.no_joint_overlap is False
    assert defaults.min_joint_neighbor_overlap is None
    off = agm.parse_args([str(tmp_path), '--no-joint-overlap'])
    tuned = agm.parse_args([str(tmp_path), '--min-joint-neighbor-overlap', '0.2'])

    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(n_per=600, seed=41)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    bins = np.linspace(-0.3, 0.3, defaults.bins + 1)
    args_kw = dict(logw=np.zeros(cv.size), bins=bins, kbt_kcal=0.6, out=tmp_path,
                   warnings=[], progress=None)
    on_info = pmfmod.run_pmf_and_gamd_boost_report(d, defaults, **args_kw)
    assert on_info['joint_overlap']['available'] is True
    assert on_info['joint_overlap']['threshold'] == pytest.approx(0.30 ** 2)
    off_info = pmfmod.run_pmf_and_gamd_boost_report(d, off, **args_kw)
    assert off_info['joint_overlap']['available'] is False
    tuned_info = pmfmod.run_pmf_and_gamd_boost_report(d, tuned, **args_kw)
    assert tuned_info['joint_overlap']['threshold'] == pytest.approx(0.2)


# ===========================================================================
# ROUND-3 FINDING 1 -- the graded set could not certify BRIDGING
#
# Two defects, one root. The A6 fix above made the health check grade the
# CV-space nearest-neighbour pairing *instead of* the index-adjacent array
# (the `else` branch that graded s['neighbor_overlap'] became dead code for
# every new analysis, since _report_summary_fields always emits
# cv_space_neighbor_overlap and cv_space_neighbor_overlap() is non-empty for
# any run with >=2 populated states). So a broken umbrella ladder could read
# PASS on data that had not changed. Fixed by grading BOTH pairings and taking
# the worse -- same worse-of-both convention the joint/marginal split already
# uses.
#
# The deeper half: NO pairwise-worst number can certify bridging at all. A
# nearest-neighbour pairing over K states contributes at most K edges (27
# states -> 19 unique pairs on chignolin_6's real matrix), so it is a spanning
# forest at best and cannot examine the wide gaps; and even a pairing where
# every graded pair clears the threshold can describe a set that is split in
# two. MBAR cannot determine the free-energy offset between two components that
# never exchange samples -- which is exactly why chignolin_6's f_23 was
# unconstrained and 94% of its posterior landed on one state. That is a
# CONNECTIVITY property of the whole overlap graph, so it gets measured
# directly.
#
# Measured on RUNS/chignolin_6/adaptive_production/adaptive_union_mbar.npz
# (27 states, 126,470 merged samples, read-only) while writing this:
#   * CV1-MARGINAL matrix at >=0.30      -> 1 component (fully connected)
#   * JOINT (CV1, CV2) matrix at >=0.09  -> 3 components:
#       [18] 3,410 samples | [20] 6,030 samples | the other 25 states 117,030
# i.e. the connectivity check has to run in the joint space to see this run's
# split at all. Computing it on the always-available marginal matrix alone
# would have re-committed, inside the fix, the very blindness the fix is for.
# (The "8 disconnected components" figure in the review request is the
# component count of the nearest-neighbour PAIRING graph, not of a thresholded
# overlap graph -- 19 edges over 27 nodes must split into 8 pieces by
# arithmetic. The conclusion drawn from it is right; the number measures the
# pairing construction, not this run's bridging.)
# ===========================================================================
def _sq_overlap(blocks, K, within=0.8, across=0.0):
    """Block-diagonal overlap matrix: ``within`` inside each block of state
    indices, ``across`` between blocks, 1.0 on the diagonal."""
    O = np.full((K, K), float(across))
    for blk in blocks:
        for i in blk:
            for j in blk:
                O[i, j] = within
    np.fill_diagonal(O, 1.0)
    return O


def test_overlap_components_is_one_component_for_a_connected_ladder():
    O = _sq_overlap([[0, 1, 2, 3]], 4, within=0.5)
    conn = pmfmod.overlap_components(O, 0.30, n_k=np.full(4, 100))
    assert conn['n_components'] == 1
    assert conn['components'] == [[0, 1, 2, 3]]
    assert conn['component_samples'] == [400]
    assert conn['threshold'] == pytest.approx(0.30)


def test_overlap_components_splits_an_isolated_state_and_counts_its_samples():
    """chignolin_6's real joint-space shape: one giant component plus isolated
    small-N states (there, [18] 3,410 and [20] 6,030 samples)."""
    O = _sq_overlap([[0, 1, 2]], 4, within=0.5)          # state 3 touches nobody
    conn = pmfmod.overlap_components(O, 0.30, n_k=np.array([100, 200, 300, 7]))
    assert conn['n_components'] == 2
    assert conn['components'] == [[0, 1, 2], [3]]
    assert conn['component_samples'] == [600, 7]
    # The per-component sample counts are what let an operator tell small-N
    # histogram deflation from a real physical gap, so they must be published.
    assert 'component_samples' in conn


def test_overlap_components_excludes_unsampled_states_from_the_graph():
    """A zero-sample state has an identically-zero overlap row, so including it
    would make every run with an unsampled window trivially disconnected -- a
    condition the dedicated 'Window sampling' check already owns."""
    O = _sq_overlap([[0, 1, 2]], 4, within=0.5)
    conn = pmfmod.overlap_components(O, 0.30, n_k=np.array([100, 200, 300, 0]))
    assert conn['n_components'] == 1
    assert conn['components'] == [[0, 1, 2]]
    assert conn['excluded_unsampled_states'] == [3]


def test_overlap_components_needs_two_graded_states():
    """Fewer than two eligible states cannot be disconnected; report 0/1 rather
    than inventing a split (and never raise -- old/partial inputs reach here)."""
    O = np.array([[1.0]])
    assert pmfmod.overlap_components(O, 0.30, n_k=np.array([5]))['n_components'] == 1
    empty = pmfmod.overlap_components(O, 0.30, n_k=np.array([0]))
    assert empty['n_components'] == 0
    assert pmfmod.overlap_components(np.zeros((0, 0)), 0.30)['n_components'] == 0
    # NaN entries must not become edges (comparison is False, not a raise).
    Onan = np.array([[1.0, np.nan], [np.nan, 1.0]])
    assert pmfmod.overlap_components(Onan, 0.30)['n_components'] == 2


def test_a_pairwise_worst_overlap_can_pass_while_the_set_is_split():
    """The argument for the whole check, in five lines: two well-overlapped
    pairs that do not touch each other. Every CV-space nearest-neighbour pair
    is 0.80 -- a perfect PASS on any worst-pair metric -- while MBAR has no
    information linking the two halves at all."""
    K = 4
    O = _sq_overlap([[0, 1], [2, 3]], K, within=0.8)
    centers = np.array([0.0, 0.02, 10.0, 10.02])
    k_kcal = np.full(K, 100.0)
    nk = np.full(K, 500)
    rows = pmfmod.cv_space_neighbor_overlap(O, centers, k_kcal, None, None, 0.6, n_k=nk)
    assert min(r['overlap'] for r in rows) == pytest.approx(0.8)   # worst pair: PASS
    conn = pmfmod.overlap_components(O, 0.30, n_k=nk)
    assert conn['n_components'] == 2                                # ... yet split
    assert conn['components'] == [[0, 1], [2, 3]]


def test_connectivity_is_measured_in_both_spaces_and_the_joint_is_the_one_that_sees_it(tmp_path):
    """The real run in miniature: states sharing one CV1 centre and ~13 sigma
    apart on CV2. The CV1-marginal graph is fully connected (which is what
    chignolin_6 reported) while the joint graph is completely split."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(n_per=800, seed=60)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    info = _report(d, tmp_path)
    conn = info['overlap_connectivity']
    assert conn['marginal']['available'] is True
    assert conn['marginal']['n_components'] == 1, conn['marginal']
    assert conn['joint']['available'] is True
    assert conn['joint']['n_components'] == 3, conn['joint']
    assert sorted(conn['joint']['components']) == [[0], [1], [2]]
    assert sum(conn['joint']['component_samples']) == cv.size
    # Each space graded against its OWN threshold, as everywhere else here.
    assert conn['marginal']['threshold'] == pytest.approx(0.30)
    assert conn['joint']['threshold'] == pytest.approx(0.09)


def test_connectivity_says_so_when_the_joint_space_was_not_computed(tmp_path):
    """--no-joint-overlap leaves the connectivity blind on the axis that failed
    on the real run; an absent joint component count must carry the reason, not
    hand back the marginal's clean bill of health silently."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(n_per=600, seed=61)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    info = _report(d, tmp_path, no_joint_overlap=True)
    conn = info['overlap_connectivity']
    assert conn['marginal']['n_components'] == 1
    assert conn['joint']['available'] is False
    assert conn['joint']['reason']
    assert 'no-joint-overlap' in conn['joint']['reason']


def test_a_healthy_2d_run_reports_a_connected_graph_in_both_spaces(tmp_path):
    """The check must be silent on a well-resolved run -- no new failure for
    anyone re-analysing a healthy campaign."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _clean_2d_run(n_per=3000, seed=62)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    warns: list = []
    info = _report(d, tmp_path, warns=warns, bins=np.linspace(-0.1, 0.3, 21))
    conn = info['overlap_connectivity']
    assert conn['marginal']['n_components'] == 1
    assert conn['joint']['n_components'] == 1, conn['joint']
    assert not any('DISCONNECTED' in w for w in warns), warns


def test_a_disconnected_overlap_graph_is_warned_and_triaged_high(tmp_path):
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(n_per=700, seed=63)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    warns: list = []
    _report(d, tmp_path, warns=warns)
    hits = [w for w in warns if 'DISCONNECTED' in w]
    assert hits, warns
    assert 'joint' in hits[0].lower()
    # Per-component sample counts are in the warning text itself: that is what
    # tells an operator whether a lone component is a real gap or small-N
    # histogram deflation.
    assert '700' in hits[0], hits[0]
    groups = gr.classify_warnings(hits)
    assert groups[0]['severity'] == 'HIGH'


def test_the_health_verdict_fails_a_disconnected_overlap_graph():
    s = _summary(overlap_connectivity={
        'marginal': {'space': 'cv1_marginal', 'available': True, 'reason': '',
                     'threshold': 0.30, 'n_components': 1, 'components': [[0, 1]],
                     'component_samples': [200]},
        'joint': {'space': 'cv1_cv2_joint', 'available': True, 'reason': '',
                  'threshold': 0.09, 'n_components': 3,
                  'components': [[0], [1], [2]], 'component_samples': [10, 20, 30]},
    })
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v['checks'] if c['name'] == 'Overlap connectivity')
    assert c['status'] == 'fail', c
    assert '3' in c['detail']
    assert v['overall'] == 'FAIL'


def test_the_health_verdict_passes_a_connected_overlap_graph():
    s = _summary(overlap_connectivity={
        'marginal': {'available': True, 'threshold': 0.30, 'n_components': 1,
                     'components': [[0, 1, 2]], 'component_samples': [300]},
        'joint': {'available': True, 'threshold': 0.09, 'n_components': 1,
                  'components': [[0, 1, 2]], 'component_samples': [300]},
    })
    v = gr.build_health_verdict(s, 0.30)
    c = next(c for c in v['checks'] if c['name'] == 'Overlap connectivity')
    assert c['status'] == 'pass', c


def test_the_health_verdict_grades_index_adjacency_as_well_as_cv_space():
    """The mirror of test_health_verdict_prefers_cv_space_overlap_over_index_
    adjacency above: the CV-space pairing says everything is fine while the
    index-adjacent ladder is broken. Before this round the CV-space number
    REPLACED the index one, so this read PASS."""
    s = _summary(cv_space_neighbor_overlap=[
        {'window': 0, 'neighbor': 1, 'overlap': 0.85},
        {'window': 1, 'neighbor': 0, 'overlap': 0.85},
    ])
    s['neighbor_overlap'] = [0.9, 0.9, 0.02] + [0.9] * 6
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'fail', ov
    assert '0.020' in ov['detail'] or '0.02' in ov['detail']
    assert '2-3' in ov['detail'], ov['detail']
    # ... and the CV-space pair is still the one named as physically adjacent.
    assert 'CV-space nearest' in ov['detail']


def test_the_index_adjacent_fold_in_cannot_downgrade_the_cv_space_status():
    """Worse-of-both, in the other direction: a healthy index ladder must not
    rescue a broken CV-space pair (the fix must not be able to weaken either
    number, which is the mistake it is undoing)."""
    s = _summary(cv_space_neighbor_overlap=[
        {'window': 0, 'neighbor': 1, 'overlap': 0.04},
        {'window': 1, 'neighbor': 0, 'overlap': 0.04},
    ])
    s['neighbor_overlap'] = [0.99] * 9
    v = gr.build_health_verdict(s, 0.30)
    ov = next(c for c in v['checks'] if c['name'] == 'Window overlap')
    assert ov['status'] == 'fail', ov
    assert ov['metric'] == pytest.approx(0.04)


def test_overlap_components_reports_each_component_s_best_escape_route():
    """How badly split, not just whether: a block sharing nothing with anything
    (undetermined free energy) and a block whose only link merely misses the
    target are the same component count and completely different diagnoses."""
    K = 4
    O = _sq_overlap([[0, 1, 2]], K, within=0.5)
    O[3, 0] = O[0, 3] = 0.07          # state 3's only link, below the 0.30 target
    O[3, 1] = O[1, 3] = 0.02
    conn = pmfmod.overlap_components(O, 0.30, n_k=np.full(K, 100))
    assert conn['n_components'] == 2
    assert conn['component_best_cross_overlap'] == pytest.approx([0.07, 0.07])
    assert conn['worst_component_best_cross_overlap'] == pytest.approx(0.07)
    # One component -> nothing across it to measure, and absence must not read
    # as a measured 0.0.
    whole = pmfmod.overlap_components(_sq_overlap([[0, 1, 2, 3]], K, within=0.5), 0.30)
    assert whole['worst_component_best_cross_overlap'] is None
    assert whole['most_isolated_component'] is None
    allnan = np.full((2, 2), np.nan)
    assert pmfmod.overlap_components(allnan, 0.30)['worst_component_best_cross_overlap'] is None


def test_the_grading_number_is_the_most_isolated_component_not_the_best_cut():
    """chignolin_6's real 3-component joint shape, with its real numbers: the
    giant block, [18] linked at 0.06481, and [20] linked at 0.00075 to the giant
    block and 0.00480 to [18] -- so the components' best escape routes are
    0.06481 / 0.06481 / 0.00480.

    The grading number must be the MINIMUM over components' best escape routes
    (0.0048, belonging to [20]), not the MAXIMUM over cut edges (0.0648). This
    test exists because the first version of this code took the max, and the
    real npz then graded the run CAUTION -- state 18's weak-but-real link talked
    state 20's total isolation down a band."""
    K = 5
    O = _sq_overlap([[0, 1, 2]], K, within=0.5)   # giant block = 0,1,2
    O[3, 0] = O[0, 3] = 0.0648                    # [18]-ish: weak but real
    O[4, 1] = O[1, 4] = 0.00075                   # [20]-ish: nothing
    O[4, 3] = O[3, 4] = 0.0048
    conn = pmfmod.overlap_components(O, 0.09, n_k=np.full(K, 100))
    assert conn['components'] == [[0, 1, 2], [3], [4]]
    assert conn['component_best_cross_overlap'] == pytest.approx([0.0648, 0.0648, 0.0048])
    assert conn['worst_component_best_cross_overlap'] == pytest.approx(0.0048)
    assert conn['most_isolated_component'] == [4]
    # ... and that is what decides the verdict band.
    conn.update({'available': True, 'space': 'cv1_cv2_joint'})
    chk = gr._check_overlap_connectivity({'overlap_connectivity': {'joint': conn}})
    assert chk['status'] == 'fail', chk        # 0.0048 < 0.5 * 0.09
    assert '[4]' in chk['detail']


def test_the_reported_isolation_depth_is_the_one_the_verdict_grades(tmp_path):
    """Report-level: the number pmf.py measures is the number gareus_report
    grades, with no re-derivation in between."""
    cv, cv2, win, centers, k_kcal, centers2, k2 = _cv1_degenerate_2d_run(n_per=700, seed=64)
    d = _make_data(cv, cv2, win, centers, k_kcal, centers2, k2, out_dir=tmp_path)
    conn = _report(d, tmp_path)['overlap_connectivity']
    best = conn['joint']['worst_component_best_cross_overlap']
    assert best is not None and best < 0.09
    chk = gr._check_overlap_connectivity({'overlap_connectivity': conn})
    assert chk['status'] == 'fail'           # ~13 sigma apart: no information
    assert f'{best:.4f}' in chk['detail']


# ===========================================================================
# M4 -- a graded overlap number must say which space it was measured in
# ===========================================================================
# `metric` changed meaning when the joint matrix arrived: joint if present,
# else the CV-space marginal, else index-adjacent. It is published in
# pmf_summary.json's health block, and the two spaces are the same shape and
# scale with different thresholds -- an external consumer grading `metric`
# against --min-neighbor-overlap would silently misread a joint value as a
# marginal one. Nothing in-repo consumes it, which is exactly why a drift here
# would be invisible.
def test_a_graded_overlap_metric_says_which_space_it_came_from():
    joint = _summary(
        cv_space_neighbor_overlap=[{'window': 0, 'neighbor': 1, 'overlap': 0.97}],
        joint_overlap={'available': True, 'threshold': 0.09, 'dim': 2,
                       'cv_space_neighbor_overlap': [
                           {'window': 0, 'neighbor': 1, 'overlap': 0.02}]})
    ov = next(c for c in gr.build_health_verdict(joint, 0.30)['checks']
              if c['name'] == 'Window overlap')
    # The joint number won, so the space must be the joint one -- 0.02 read as
    # a marginal overlap is a catastrophe, read as a joint one it is merely bad.
    assert ov['metric'] == pytest.approx(0.02)
    assert ov['metric_space'] == gr.OVERLAP_SPACE_JOINT

    marginal = _summary(
        cv_space_neighbor_overlap=[{'window': 0, 'neighbor': 1, 'overlap': 0.42}],
        joint_overlap={'available': False, 'reason': 'no secondary restraint'})
    ov = next(c for c in gr.build_health_verdict(marginal, 0.30)['checks']
              if c['name'] == 'Window overlap')
    assert ov['metric'] == pytest.approx(0.42)
    assert ov['metric_space'] == gr.OVERLAP_SPACE_MARGINAL


def test_the_index_adjacent_fallback_metric_is_also_labelled():
    """The oldest summary format: no CV-space list, no joint block at all."""
    s = _summary()
    s['neighbor_overlap'] = [0.5] * 8 + [0.11]
    ov = next(c for c in gr.build_health_verdict(s, 0.30)['checks']
              if c['name'] == 'Window overlap')
    assert ov['metric'] == pytest.approx(0.11)
    assert ov['metric_space'] == gr.OVERLAP_SPACE_MARGINAL


def test_metric_space_uses_the_analysis_module_s_own_vocabulary():
    """gareus_report duplicates these two strings to stay standalone-importable.

    Duplication only stays safe while it is checked: if pmf.py ever renames a
    space, a consumer comparing a check's ``metric_space`` with the summary's
    own ``overlap_space`` would start silently matching nothing.
    """
    assert gr.OVERLAP_SPACE_MARGINAL == pmfmod.OVERLAP_SPACE_MARGINAL
    assert gr.OVERLAP_SPACE_JOINT == pmfmod.OVERLAP_SPACE_JOINT
    assert gr.OVERLAP_SPACE_MARGINAL != gr.OVERLAP_SPACE_JOINT


def test_an_ungraded_overlap_check_carries_neither_metric_nor_space():
    s = _summary()
    s['neighbor_overlap'] = []
    ov = next(c for c in gr.build_health_verdict(s, 0.30)['checks']
              if c['name'] == 'Window overlap')
    assert ov['status'] == 'na'
    assert 'metric' not in ov and 'metric_space' not in ov

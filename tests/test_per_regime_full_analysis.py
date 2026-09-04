"""Tests for analysing a CV2-regime-changing run as N independent populations.

``analyze`` used to be one flat pass: a single pooled MBAR solve over every
sample, with the CV2-facing plots split per regime afterwards. Across a CV2
regime change that pooled solve is not a solve of any Hamiltonian -- each row
block is evaluated against its own regime's cv2, so a state's column mixes two
bias definitions, and states created after the switch have no meaningful value
on pre-switch rows.

``analyze`` now detects the regimes and runs the FULL analysis once per
regime, each with its own solve and its own output tree. These tests pin the
orchestration: that a single-regime run is untouched (the case most likely to
regress, and the overwhelming majority of runs), that two regimes produce two
populations in the right directories, and that convergence is skipped for the
non-dominant one in a way that records why.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import analyze_gareus_mbar as agm  # noqa: E402
from gareus.mbar_analysis.data import Data  # noqa: E402
from gareus.units import K_B_KJ_PER_MOL_K  # noqa: E402

BETA = 1.0 / (K_B_KJ_PER_MOL_K * 300.0)


def _harmonic_block(rng, centers, k_kj, n_per, step_offset, beta):
    """Samples from a 1-D harmonic umbrella ladder, plus its u_nk."""
    cv, window = [], []
    for w, c in enumerate(centers):
        sigma = float(np.sqrt(1.0 / (beta * k_kj[w])))
        cv.append(rng.normal(c, sigma, n_per[w]))
        window.append(np.full(n_per[w], w, dtype=np.int64))
    cv = np.concatenate(cv)
    window = np.concatenate(window)
    u_nk = np.empty((cv.size, len(centers)))
    for w, c in enumerate(centers):
        u_nk[:, w] = beta * 0.5 * k_kj[w] * (cv - c) ** 2
    step = np.arange(cv.size, dtype=np.int64) + step_offset
    return cv, window, u_nk, step


def _make_data(tmp_path, regimes, n_per=(300, 300), seed=5):
    """A Data whose phases record the given secondary_cv modes.

    ``regimes`` is a list of mode names, one per epoch dir. One entry gives a
    single-regime run (the mask builder returns None); two or more give a
    regime-changing run.
    """
    rng = np.random.default_rng(seed)
    centers = [0.0, 1.0]
    k_kj = [800.0, 800.0]
    cvs, windows, unks, steps, srcs = [], [], [], [], []
    run_dirs = []
    offset = 0
    for i, regime in enumerate(regimes):
        cv, win, u_nk, step = _harmonic_block(rng, centers, k_kj, list(n_per), offset, BETA)
        offset += cv.size
        cvs.append(cv); windows.append(win); unks.append(u_nk); steps.append(step)
        srcs.extend([i] * cv.size)
        ed = tmp_path / f'epoch_{i:03d}'
        ed.mkdir(parents=True, exist_ok=True)
        (ed / 'run_manifest.json').write_text(
            json.dumps({'resolved_args': {'secondary_cv': regime}}))
        run_dirs.append(str(ed))
    cv = np.concatenate(cvs)
    n = cv.size
    out_dir = tmp_path / 'pmf_analysis'
    return Data(
        prod_dir=tmp_path, out_dir=out_dir,
        cv=cv, cv2=rng.normal(0.0, 1.0, n), rg_A=np.full(n, np.nan),
        window=np.concatenate(windows), replica=np.zeros(n, dtype=np.int64),
        step=np.concatenate(steps), u_nk=np.concatenate(unks, axis=0),
        centers=np.asarray(centers), k_kcal=np.asarray(k_kj) / 4.184,
        beta=BETA, temp=300.0, boost_kj=np.zeros(n), potential_kj=None,
        source='synthetic',
        meta={'_epoch_source': srcs, 'adaptive_epoch_run_dirs': run_dirs},
    )


class _Recorder:
    """Stands in for _analyze_population, recording how it was called."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, d, args, out, progress=None, *, regime=None,
                 run_convergence=True, convergence_skip_reason=''):
        self.calls.append({'out': Path(out), 'regime': regime,
                           'run_convergence': run_convergence,
                           'skip_reason': convergence_skip_reason,
                           'n_samples': int(d.cv.size),
                           'd_out_dir': Path(d.out_dir)})
        return {'output_dir': str(out), 'secondary_cv_regime': regime,
                'n_samples': int(d.cv.size), 'mbar': {'converged': True},
                'warnings': []}


def _args():
    a = agm.parse_args(['dummy', '--bins', '12'])
    a.no_poincare_map = True
    return a


# --- the single-regime path must be untouched -------------------------------

def test_single_regime_delegates_once_with_the_datas_own_out_dir(tmp_path, monkeypatch):
    """The case most likely to regress and by far the most common. The regime
    loop must not engage at all: one call, the Data's own out_dir, convergence
    on, and no regime label."""
    d = _make_data(tmp_path, ['torsion-pca'])
    rec = _Recorder()
    monkeypatch.setattr(agm, '_analyze_population', rec)

    out = agm.analyze(d, _args())

    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call['out'] == Path(d.out_dir)
    assert call['regime'] is None
    assert call['run_convergence'] is True
    assert call['n_samples'] == int(d.cv.size)
    # No regime index is attached to a single-regime summary.
    assert 'regime_analyses' not in out


def test_run_without_epoch_metadata_is_also_single_pass(tmp_path, monkeypatch):
    """No _epoch_source means the mask builder cannot group anything; that must
    degrade to the plain single-pass analysis, not to zero passes."""
    d = _make_data(tmp_path, ['torsion-pca'])
    d.meta = {}
    rec = _Recorder()
    monkeypatch.setattr(agm, '_analyze_population', rec)
    agm.analyze(d, _args())
    assert len(rec.calls) == 1
    assert rec.calls[0]['regime'] is None


# --- two regimes -> two populations -----------------------------------------

def test_two_regimes_get_their_own_solves_and_output_trees(tmp_path, monkeypatch):
    d = _make_data(tmp_path, ['torsion-pca', 'tica-linear'])
    rec = _Recorder()
    monkeypatch.setattr(agm, '_analyze_population', rec)

    out = agm.analyze(d, _args())

    assert len(rec.calls) == 2, 'expected one population per regime'
    by_regime = {c['regime']: c for c in rec.calls}
    assert set(by_regime) == {'torsion-pca', 'tica-linear'}

    # Dominant regime (the one holding the LAST phase) writes to the root, so
    # existing paths keep working; the other gets its own subtree.
    dominant = out['secondary_cv_regime']
    assert dominant == 'tica-linear'
    assert by_regime['tica-linear']['out'] == Path(d.out_dir)
    assert by_regime['torsion-pca']['out'] == Path(d.out_dir) / 'regime_torsion-pca'

    # Each population sees only its own samples.
    assert by_regime['torsion-pca']['n_samples'] == 600
    assert by_regime['tica-linear']['n_samples'] == 600

    # The Data handed to each population also has its out_dir retargeted, so a
    # path derived from d.out_dir rather than the `out` argument cannot leak
    # one regime's files into the other's tree.
    for regime, call in by_regime.items():
        assert call['d_out_dir'] == call['out'], regime


def test_non_dominant_regime_gets_no_convergence_and_says_why(tmp_path, monkeypatch):
    d = _make_data(tmp_path, ['torsion-pca', 'tica-linear'])
    rec = _Recorder()
    monkeypatch.setattr(agm, '_analyze_population', rec)
    agm.analyze(d, _args())

    by_regime = {c['regime']: c for c in rec.calls}
    assert by_regime['tica-linear']['run_convergence'] is True
    assert by_regime['torsion-pca']['run_convergence'] is False
    # An absent stage must carry its reason, and name the way to turn it on.
    reason = by_regime['torsion-pca']['skip_reason']
    assert 'non-dominant' in reason
    assert '--convergence-all-regimes' in reason
    assert by_regime['tica-linear']['skip_reason'] == ''


def test_convergence_all_regimes_flag_turns_it_back_on(tmp_path, monkeypatch):
    d = _make_data(tmp_path, ['torsion-pca', 'tica-linear'])
    rec = _Recorder()
    monkeypatch.setattr(agm, '_analyze_population', rec)
    args = _args()
    args.convergence_all_regimes = True
    agm.analyze(d, args)
    assert all(c['run_convergence'] is True for c in rec.calls)


def test_dominant_summary_indexes_every_regime_and_warns_about_its_scope(tmp_path, monkeypatch):
    """The top-level summary is now ONE regime's, not the run's. It must say so
    loudly: reading a minority regime's numbers as the whole run is the
    highest-risk silent misread this split introduces."""
    d = _make_data(tmp_path, ['torsion-pca', 'tica-linear'])
    monkeypatch.setattr(agm, '_analyze_population', _Recorder())
    out = agm.analyze(d, _args())

    idx = out['regime_analyses']
    assert set(idx) == {'torsion-pca', 'tica-linear'}
    assert idx['tica-linear']['is_dominant'] is True
    assert idx['torsion-pca']['is_dominant'] is False
    assert idx['torsion-pca']['convergence_ran'] is False
    assert idx['tica-linear']['convergence_ran'] is True
    for entry in idx.values():
        assert entry['summary_json'].endswith('pmf_summary.json')
        assert entry['n_samples'] == 600

    warning = ' '.join(out['warnings'])
    assert 'redefined its secondary CV' in warning
    assert 'only the dominant regime' in warning
    assert 'not comparable' in warning


# --- end to end -------------------------------------------------------------

@pytest.mark.parametrize('regimes,expect_split', [
    (['torsion-pca'], False),
    (['torsion-pca', 'tica-linear'], True),
])
def test_end_to_end_writes_one_summary_per_population(tmp_path, regimes, expect_split):
    """A real analyze() run, no monkeypatching: the number of pmf_summary.json
    files on disk must equal the number of populations, and a skipped
    convergence stage must be recorded as skipped rather than merely absent."""
    d = _make_data(tmp_path, regimes, n_per=(220, 220), seed=11)
    args = _args()
    args.no_extra_pmfs = True
    args.no_convergence = True          # keep the run fast; the SKIP path is
    args.no_basin_tracking = True       # asserted from the summary below
    s = agm.analyze(d, args)

    root_summary = Path(d.out_dir) / 'pmf_summary.json'
    assert root_summary.is_file(), 'dominant regime must write to the root'
    written = sorted(p for p in Path(d.out_dir).rglob('pmf_summary.json'))
    assert len(written) == len(regimes), [str(p) for p in written]

    if not expect_split:
        assert s.get('secondary_cv_regime') is None
        assert 'regime_analyses' not in s
        assert not (Path(d.out_dir) / 'regime_torsion-pca').exists()
        return

    assert s['secondary_cv_regime'] == 'tica-linear'
    sub = Path(d.out_dir) / 'regime_torsion-pca' / 'pmf_summary.json'
    assert sub.is_file()
    sub_s = json.loads(sub.read_text())
    assert sub_s['secondary_cv_regime'] == 'torsion-pca'
    # Skipped, and self-describing: a consumer must not have to infer whether
    # a missing convergence block means "deliberately skipped" or "crashed".
    conv = sub_s['convergence']
    assert conv.get('skipped') is True
    assert 'non-dominant' in conv.get('reason', '')
    # Each population solved MBAR on its own rows only.
    assert sub_s['n_samples'] == 440
    assert json.loads(root_summary.read_text())['n_samples'] == 440

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import analyze_gareus_mbar as agm
from gareus.mbar_analysis import pmf


def _data(z, *, ladder=False):
    n = len(z)
    return SimpleNamespace(
        aux_z=np.asarray(z), cv=np.zeros(n), u_nk=np.zeros((n, 2)),
        boost_kj=np.zeros(n), beta=1.0, temp=300.0,
        meta={'aux_models': ['model-sha'], 'aux_model_sha256': 'model-sha',
              'gamd_ladder': ladder},
    )


def _analyze(d, tmp_path, logw=None, selected='umbrella_only', boost_ok=False):
    args = agm.parse_args(['dummy', '--bins', '2'])
    return pmf.analyze_auxiliary_cv_pmf(
        d, args, np.zeros(len(d.cv)) if logw is None else logw,
        selected, boost_ok, 0.6, tmp_path, [],
    )


def test_aux_pmf_uses_target_weights_and_writes_plot(tmp_path):
    d = _data(np.r_[np.full(20, -1.0), np.full(20, 1.0), np.nan])
    logw = np.r_[np.zeros(20), np.full(20, np.log(3.0)), 100.0]
    info = _analyze(d, tmp_path, logw)
    assert info['available'] and info['n_samples'] == 40
    assert info['model_sha256'] == 'model-sha'
    rows = list(csv.DictReader((tmp_path / 'cvaux_pmf_unbiased.csv').open()))
    np.testing.assert_allclose([float(r['probability']) for r in rows], [0.25, 0.75])
    np.testing.assert_allclose([float(r['pmf_kcal_mol']) for r in rows], [0.6 * np.log(3), 0])
    assert 'cvaux_z' in rows[0] and 'cv_A' not in rows[0]
    assert (tmp_path / 'cvaux_pmf_unbiased.png').stat().st_size > 0
    assert (tmp_path / 'cvaux_pmf_summary.json').is_file()


def test_aux_pmf_ladder_does_not_reapply_gamd_boost(tmp_path):
    d = _data(np.r_[np.full(20, -1.0), np.full(20, 1.0)], ladder=True)
    d.boost_kj = np.r_[np.zeros(20), np.full(20, 50.0)]
    info = _analyze(d, tmp_path, selected='gamd_exponential', boost_ok=True)
    assert info['selected_unbiased_method'] == 'umbrella_only'
    rows = list(csv.DictReader((tmp_path / 'cvaux_pmf_unbiased.csv').open()))
    np.testing.assert_allclose([float(r['probability']) for r in rows], [0.5, 0.5])


def test_aux_pmf_accepts_single_column(tmp_path):
    d = _data(np.linspace(-1, 1, 40)[:, None])
    assert _analyze(d, tmp_path)['available']


def test_aux_pmf_preserves_selected_umbrella_method(tmp_path):
    d = _data(np.r_[np.full(20, -1.0), np.full(20, 1.0)])
    d.boost_kj = np.r_[np.zeros(20), np.full(20, 50.0)]
    info = _analyze(d, tmp_path, selected='umbrella_only', boost_ok=True)
    assert info['selected_unbiased_method'] == 'umbrella_only'


def test_aux_pmf_selected_exponential_uses_boost_weights(tmp_path):
    d = _data(np.r_[np.full(20, -1.0), np.full(20, 1.0)])
    d.boost_kj = np.r_[np.zeros(20), np.full(20, np.log(3.0))]
    info = _analyze(d, tmp_path, selected='gamd_exponential', boost_ok=True)
    assert info['selected_unbiased_method'] == 'gamd_exponential'
    rows = list(csv.DictReader((tmp_path / 'cvaux_pmf_unbiased.csv').open()))
    np.testing.assert_allclose([float(r['probability']) for r in rows], [0.25, 0.75])


def test_aux_pmf_refuses_misaligned_coordinates(tmp_path):
    d = _data(np.linspace(-1, 1, 40))
    d.aux_z = d.aux_z[:-1]
    with pytest.raises(ValueError, match='aligned'):
        _analyze(d, tmp_path)


def test_aux_pmf_missing_samples_reports_unavailable(tmp_path):
    info = _analyze(_data(np.full(40, np.nan)), tmp_path)
    assert not info['available'] and info['n_samples'] == 0
    assert not (tmp_path / 'cvaux_pmf_unbiased.png').exists()


def test_aux_pmf_legacy_creates_no_outputs(tmp_path):
    d = _data(np.linspace(-1, 1, 40))
    d.meta = {}
    assert not _analyze(d, tmp_path)['available']
    assert list(tmp_path.iterdir()) == []

import numpy as np
import pytest

from gareus.adaptive.aux_backfill import worker_energy_error_kt, R_KCAL_MOL_K


def test_empty_neighbourhood_is_unavailable_not_zero_error():
    sigma = np.sqrt(R_KCAL_MOL_K*300.)
    row = worker_energy_error_kt([3*sigma], [4*sigma], [(0., 1.)])[0]
    assert row['n_within_2sigma'] == 0
    assert row['max_energy_err_kt'] is None
    assert row['status'] == 'insufficient_worker_coverage'
    assert row['max_energy_err_kt_all'] == pytest.approx(3.5)


def test_covered_matching_data():
    row = worker_energy_error_kt([0., .1], [0., .1], [(0., 1.)])[0]
    assert row['status'] == 'ok'
    assert row['max_energy_err_kt'] == 0.
    assert row['n_compared'] == 2


@pytest.mark.parametrize('left,right', [([], []), ([np.nan], [0]), ([0], [np.inf]), ([0,1],[0])])
def test_invalid_data_refused(left,right):
    with pytest.raises(ValueError):
        worker_energy_error_kt(left,right,[(0.,1.)])


def test_real_spot_check_refuses_empty_coverage(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import gareus.adaptive.aux_backfill as backfill
    sigma = np.sqrt(R_KCAL_MOL_K*300.)
    samples = tmp_path/'samples'/'replica_0'
    samples.mkdir(parents=True)
    pd.DataFrame({'replica':[0], 'step':[10], 'aux_z_00':[3*sigma]}).to_parquet(samples/'seg_0.parquet')
    monkeypatch.setattr(backfill, 'frame_z', lambda *a,**kw: pd.DataFrame(
        {'replica':[0], 'step':[10], 'aux_z':[4*sigma]}))
    result = backfill.check_backfill_against_recorded(tmp_path, None, workers=[(0.,1.)])
    assert result['ok'] is False
    assert result['status'] == 'insufficient_worker_coverage'
    assert result['max_energy_err_kt'] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='insufficient_worker_coverage'):
        backfill.check_backfill_against_recorded(tmp_path, None, workers=[(0.,1.)], raise_on_fail=True)

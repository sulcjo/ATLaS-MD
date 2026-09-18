"""steps-per-frame must resolve from the run config, and never from a guessable default.

chignolin_7 lost 98% of its coordinate data here. `_read_traj_interval` searched
only `d.prod_dir`, which for an adaptive run is `<run>/adaptive_production` -- but
`traj_interval` is written at the RUN ROOT, one level up. Nothing was found, so it
returned its default of 50 against a real value of 2500.

The default is what made it silent. Every call site guarded with

    spf = _read_traj_interval(d.prod_dir)
    if spf <= 0 or spf == 500:          # 500 is the OTHER function's default
        spf = _read_traj_interval_from_epoch_dirs(d)

and 50 is neither <= 0 nor == 500, so the epoch-dir fallback never ran. An
unresolved value was indistinguishable from a configured one.

Downstream, `_sample_to_segment_frame` accepts samples in
[resume_start + spf, resume_start + n_frames*spf]. Shrinking spf by 50x shrinks
that window by 50x, so ~1/50 of samples matched: measured 103,664/5,124,128 =
2.023% against 2.00% predicted. The rest were dropped by a bare `continue`.

Same shape as the gareus_report packaging bug fixed the same day: a
wrong-but-plausible fallback standing in for an unresolved value, with no signal.
"""
import json

import pytest

import analyze_gareus_mbar as A


def _write_cfg(d, traj_interval):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'run_args.json').write_text(json.dumps({'traj_interval': traj_interval}))


def test_resolves_from_the_run_root_when_prod_dir_has_no_config(tmp_path):
    """The exact failing layout: config at the run root, prod_dir one level down."""
    run_root = tmp_path / 'chignolin_x'
    prod = run_root / 'adaptive_production'
    prod.mkdir(parents=True)
    _write_cfg(run_root, 2500)
    assert A._read_traj_interval_one_dir(prod) == 0      # genuinely absent there
    assert A._read_traj_interval(prod) == 2500           # ...but found by the walk


def test_unresolved_returns_zero_not_a_plausible_step_count():
    """0 is the whole point: a caller can tell 'not found' from 'configured'.
    Returning 50 is what disabled every guard."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        deep = Path(td) / 'a' / 'b' / 'c'
        deep.mkdir(parents=True)
        assert A._read_traj_interval(deep) == 0


def test_prod_dir_config_wins_over_the_parent(tmp_path):
    run_root = tmp_path / 'run'
    prod = run_root / 'adaptive_production'
    _write_cfg(run_root, 2500)
    _write_cfg(prod, 250)
    assert A._read_traj_interval(prod) == 250


class _D:
    """Minimal stand-in for Data: the resolver only touches prod_dir and meta."""
    def __init__(self, prod_dir, meta=None):
        self.prod_dir = prod_dir
        self.meta = meta if meta is not None else {}


def test_resolver_reports_its_source(tmp_path):
    run_root = tmp_path / 'run'
    prod = run_root / 'adaptive_production'
    prod.mkdir(parents=True)
    _write_cfg(run_root, 2500)
    spf, src = A._resolve_traj_interval(_D(prod))
    assert spf == 2500
    assert 'config' in src


def test_resolver_warns_loudly_when_nothing_resolves(tmp_path):
    """The last-resort default still exists, but it can no longer be silent."""
    prod = tmp_path / 'run' / 'adaptive_production'
    prod.mkdir(parents=True)
    warns: list = []
    spf, src = A._resolve_traj_interval(_D(prod), warns)
    assert spf == A._TRAJ_INTERVAL_LAST_RESORT
    assert 'UNRESOLVED' in src.upper()
    assert warns and 'traj_interval could not be resolved' in warns[0]


def test_no_call_site_reimplements_the_resolution():
    """The five open-coded copies of the guard are what let one drift. Reading
    the interval directly from d.prod_dir outside the resolver is the bug."""
    import inspect
    src = inspect.getsource(A)
    body = src.split('def _resolve_traj_interval', 1)[1]
    after = body.split('\ndef ', 2)
    rest = '\ndef '.join(after[2:]) if len(after) > 2 else ''
    assert '_read_traj_interval(d.prod_dir)' not in rest, (
        'a call site is resolving traj_interval itself instead of using '
        '_resolve_traj_interval(); that is how the 50-vs-500 sentinel mismatch survived'
    )


@pytest.mark.parametrize('spf,expected_matches', [(2500, 4), (50, 0)])
def test_wrong_spf_shrinks_the_acceptance_window(spf, expected_matches):
    """Mechanism check: the window is [R+spf, R+n_frames*spf], so an spf that is
    too small silently excludes samples rather than mismapping them."""
    import numpy as np
    # 4 samples on the true 2500-step grid, one segment of 4 frames from step 0
    steps = np.array([2500, 5000, 7500, 10000], dtype=np.float64)
    mask, local = A._sample_to_segment_frame(steps, 0, 4, spf)
    assert int(mask.sum()) == expected_matches


def test_coverage_threshold_is_defined_and_sane():
    assert 0.0 < A.TRAJ_COVERAGE_MIN_FRACTION < 1.0

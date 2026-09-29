from pathlib import Path

import numpy as np
import pytest

from gareus.mbar_analysis import data, ladder, solvers
from gareus.pep_gamd import PepGamdEnvelope
from test_mbar_analysis_data_part_b import _mk_data


def mapped(tmp_path, values):
    out = np.memmap(tmp_path / 'matrix.dat', mode='w+', dtype=np.float64,
                    shape=values.shape)
    out[:] = values
    return out


def test_release_pages_preserves_copy_on_write_changes(tmp_path):
    from gareus.mbar_analysis import storage
    mapped(tmp_path, np.arange(6.).reshape(3, 2)).flush()
    private = np.memmap(tmp_path / 'matrix.dat', mode='c', dtype=np.float64, shape=(3, 2))
    private[:] = -1.
    storage.release_pages(private)
    np.testing.assert_array_equal(private, np.full((3, 2), -1.))


@pytest.mark.parametrize('operation', ['clean', 'skip', 'stride', 'subset'])
def test_row_filters_preserve_disk_backing_and_alignment(tmp_path, operation):
    d = _mk_data(n=8, meta={'_epoch_source': list(range(8))})
    d.u_nk = mapped(tmp_path, np.arange(16.).reshape(8, 2))
    if operation == 'clean':
        d.u_nk[2, 1] = np.nan
        result = data.clean(d)
        rows = [0, 1, 3, 4, 5, 6, 7]
    elif operation == 'skip':
        result = data._skip_first_n_frames(d, 1)
        rows = list(range(2, 8))
    elif operation == 'stride':
        d.meta['_epoch_source'] = [0] * 8
        result = data._apply_analysis_stride(d, 2)
        rows = [0, 1, 4, 5]
    else:
        result = data._masked_data(d, np.arange(8) % 2 == 0)
        rows = [0, 2, 4, 6]
    assert isinstance(result.u_nk, np.memmap)
    np.testing.assert_array_equal(result.cv, rows)
    np.testing.assert_array_equal(result.u_nk, np.arange(16.).reshape(8, 2)[rows])
    assert len(result.meta['_epoch_source']) == len(rows)


def test_ladder_corrects_writable_memmap_in_bounded_blocks(tmp_path, monkeypatch):
    import gareus.pep_gamd as gamd
    from gareus.mbar_analysis import storage

    monkeypatch.setattr(storage, 'BLOCK_ELEMENTS', 6)
    env = PepGamdEnvelope(50., -50., 50., .8, 50., -50., 50., .6)
    pep = np.array([-20., 0., 5., 20., 40.])
    dih = np.array([-5., 1., 3., np.nan, 30.])
    lambdas = np.array([0., .5, 1.])
    original = np.arange(15.).reshape(5, 3)
    expected = ladder.apply_ladder_boost_to_u(original, pep, dih, lambdas, env, .4, {})
    real_boost = gamd.pep_gamd_boost_matrix_kj

    def bounded_boost(p, d, ls, e):
        assert len(p) <= 2, 'full boost matrix allocated'
        return real_boost(p, d, ls, e)

    monkeypatch.setattr(gamd, 'pep_gamd_boost_matrix_kj', bounded_boost)
    u = storage.create_matrix(original.shape, directory=tmp_path)
    u[:] = original
    meta = {}
    result = ladder.apply_ladder_boost_to_u(u, pep, dih, lambdas, env, .4, meta)
    assert result is u
    np.testing.assert_allclose(result, expected)
    assert meta['gamd_ladder_samples_without_raw_energies'] == 1


@pytest.mark.parametrize('backend', ['numpy', 'anderson', 'lbfgs', 'sambar', 'numba'])
def test_solvers_bound_matrix_work_with_inactive_states(tmp_path, monkeypatch, backend):
    from gareus.mbar_analysis import storage

    if backend == 'numba' and not solvers.NUMBA_AVAILABLE:
        pytest.skip('Numba unavailable')
    if backend == 'lbfgs' and not solvers.SCIPY_AVAILABLE:
        pytest.skip('SciPy unavailable')
    values = np.array([[0., .2, .5], [.1, .3, .4], [.4, .1, 0.], [.5, .2, .1]])
    window = np.array([0, 0, 2, 2])
    kw = dict(backend=backend, tol=1e-9, maxiter=100, threads=2,
              sambar_epochs=2, sambar_initial_batch_size=4,
              sambar_polish_backend='numpy')
    expected = solvers.solve_mbar(values, window, **kw)
    monkeypatch.setattr(storage, 'BLOCK_ELEMENTS', 3)
    real_empty = np.empty
    real_empty_like = np.empty_like

    def bounded_empty(shape, *args, **kwargs):
        assert not isinstance(shape, tuple) or len(shape) != 2 or shape[0] <= 1
        return real_empty(shape, *args, **kwargs)

    def bounded_empty_like(a, *args, **kwargs):
        assert a.ndim != 2 or a.shape[0] <= 1
        return real_empty_like(a, *args, **kwargs)

    monkeypatch.setattr(np, 'empty', bounded_empty)
    monkeypatch.setattr(np, 'empty_like', bounded_empty_like)
    result = solvers.solve_mbar(mapped(tmp_path, values), window, **kw)
    np.testing.assert_allclose(result['f_k'], expected['f_k'], atol=1e-8)
    np.testing.assert_allclose(result['logw'], expected['logw'], atol=1e-8)
    np.testing.assert_array_equal(result['n_k'], [2, 0, 2])
    assert result['backend'] == expected['backend']


def test_overlap_bounds_work_and_handles_inactive_nan_offsets(tmp_path, monkeypatch):
    from gareus.mbar_analysis import storage

    monkeypatch.setattr(storage, 'BLOCK_ELEMENTS', 3)
    values = np.zeros((4, 3))
    overlap = ladder.mbar_state_overlap(mapped(tmp_path, values),
                                       np.array([0., np.nan, 0.]),
                                       np.array([2., 0., 2.]))
    np.testing.assert_allclose(overlap, [[.5, 0., .5], [0., 0., 0.], [.5, 0., .5]])


def test_npz_loader_streams_bias_member_without_array_access(tmp_path, monkeypatch):
    from gareus.mbar_analysis.loaders_adaptive import load_union_npz

    np.savez_compressed(tmp_path / 'adaptive_union_mbar.npz',
                        cv_A=[1., 2.], secondary_cv=[0., 0.],
                        sampled_state_ids=[4, 8], state_ids=[4, 8],
                        umbrella_reduced_bias_nk=np.array([[0., 1.], [1., 0.]]),
                        primary_centers=[1., 2.], primary_k=[2., 2.])
    (tmp_path / 'adaptive_union_mbar.json').write_text('{"beta_1_over_kJ_mol": 0.4}')
    cls = np.lib.npyio.NpzFile
    original = cls.__getitem__

    def vectors_only(self, key):
        assert key != 'umbrella_reduced_bias_nk', 'bias member decompressed into RAM'
        return original(self, key)

    monkeypatch.setattr(cls, '__getitem__', vectors_only)
    d = load_union_npz(tmp_path, low_memory=True)
    assert isinstance(d.u_nk, np.memmap)
    np.testing.assert_array_equal(d.u_nk, [[0., 1.], [1., 0.]])
    np.testing.assert_array_equal(d.window, [0, 1])


def test_bias_reconstruction_bounds_strict_kernel_rows(monkeypatch):
    from gareus.mbar_analysis import bias, storage
    from gareus import query

    monkeypatch.setattr(storage, 'BLOCK_ELEMENTS', 4)
    original = query.reconstruct_bias_matrix

    def bounded(cv, cv2, windows, beta):
        assert len(cv) <= 2, 'whole phase reconstructed in RAM'
        return original(cv, cv2, windows, beta)

    monkeypatch.setattr(query, 'reconstruct_bias_matrix', bounded)
    u = bias._reconstruct_union_bias_block(np.arange(5.), np.zeros(5), .5,
                                           np.array([0., 1.]), np.array([2., 2.]),
                                           np.zeros(2), np.zeros(2), low_memory=True)
    assert isinstance(u, np.memmap)
    np.testing.assert_allclose(u, 2.092 * (np.arange(5.)[:, None] - [0., 1.]) ** 2)


def test_temporary_matrix_cleanup_on_reconstruction_failure(monkeypatch):
    from gareus.mbar_analysis import bias, storage
    from gareus import query

    before = set(storage._paths)
    monkeypatch.setattr(query, 'reconstruct_bias_matrix',
                        lambda *args: (_ for _ in ()).throw(ValueError('injected failure')))
    with pytest.raises(ValueError, match='injected failure'):
        bias._reconstruct_union_bias_block(np.arange(5.), np.zeros(5), .5,
                                           np.zeros(2), np.ones(2), np.zeros(2),
                                           np.zeros(2), low_memory=True)
    assert storage._paths == before


def test_cli_skips_before_stride_without_changing_selected_rows(tmp_path, monkeypatch):
    import analyze_gareus_mbar as cli

    d = _mk_data(n=8, meta={'_epoch_source': [0] * 8})
    d.u_nk = mapped(tmp_path, np.zeros((8, 2)))

    def loader(*args, **kwargs):
        assert kwargs['analysis_stride'] == 1, 'stride applied before burn-in'
        return d

    def analyzed(selected, *args, **kwargs):
        np.testing.assert_array_equal(selected.step, [2, 3, 6, 7])
        assert isinstance(selected.u_nk, np.memmap)
        raise RuntimeError('analysis reached')

    monkeypatch.setattr(cli, 'load_data', loader)
    monkeypatch.setattr(cli, 'analyze', analyzed)
    with pytest.raises(RuntimeError, match='analysis reached'):
        cli.main(['run', '--low-memory', '--skip-first-n-frames', '1', '--analysis-stride', '2'])


def test_cli_no_stride_or_skip_reaches_analysis(tmp_path, monkeypatch):
    import analyze_gareus_mbar as cli

    d = _mk_data(n=4)
    monkeypatch.setattr(cli, 'load_data', lambda *a, **kw: d)
    monkeypatch.setattr(cli, 'analyze', lambda *a, **kw:
                        (_ for _ in ()).throw(RuntimeError('analysis reached')))
    with pytest.raises(RuntimeError, match='analysis reached'):
        cli.main(['run', '--low-memory'])


@pytest.mark.parametrize('dtype,fortran', [(np.float32, False), (np.float64, True)])
def test_npz_storage_converts_dtype_and_order_in_blocks(tmp_path, dtype, fortran):
    from gareus.mbar_analysis.storage import extract_npz_matrix

    values = np.array([[0., 1.], [2., 3.]], dtype=dtype, order='F' if fortran else 'C')
    path = tmp_path / 'archive.npz'
    np.savez_compressed(path, bias=values)
    with np.load(path) as f:
        result = extract_npz_matrix(f, 'bias', directory=tmp_path)
    assert isinstance(result, np.memmap)
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, [[0., 1.], [2., 3.]])


def test_loader_failure_cleans_previous_epoch_matrices(tmp_path, monkeypatch):
    import json
    import analyze_gareus_mbar as cli
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union
    from test_loader_masked_cv2_nan import _write_epoch, _write_registry

    adaptive = tmp_path / 'adaptive_production'
    adaptive.mkdir()
    (adaptive.parent / 'run_args.json').write_text(json.dumps({'temperature_k': 300.}))
    _write_registry(adaptive, [{'state_id': '0', 'primary_center': '0.', 'primary_k': '44.3',
                                'secondary_center': '0.', 'secondary_k': '0.'}])
    _write_epoch(adaptive / 'epoch_000', [(1., 0.)], 0., 0.)
    _write_epoch(adaptive / 'epoch_001', [(2., 0.)], 0., 0.)
    original = cli._reconstruct_union_bias_block
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError('second epoch failed')
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, '_reconstruct_union_bias_block', fail_second)
    with pytest.raises(ValueError, match='second epoch failed'):
        load_parquet_adaptive_union(adaptive, low_memory=True)
    assert not list(adaptive.glob('.gareus-matrix-*'))


def test_disk_reservation_failure_cleans_temporary_file(tmp_path, monkeypatch):
    import errno
    import os
    from gareus.mbar_analysis.storage import create_matrix

    def no_space(*args):
        raise OSError(errno.ENOSPC, 'injected disk full')

    monkeypatch.setattr(os, 'posix_fallocate', no_space)
    with pytest.raises(OSError, match='injected disk full'):
        create_matrix((2, 2), directory=tmp_path)
    assert not list(tmp_path.glob('.gareus-matrix-*'))


def test_solver_does_not_cast_entire_fortran_memmap_into_ram(tmp_path, monkeypatch):
    from gareus.mbar_analysis.storage import disk_array

    u = np.lib.format.open_memmap(tmp_path / 'fortran.npy', mode='w+', dtype=np.float64,
                                  shape=(4, 2), fortran_order=True)
    u[:] = [[0., 1.], [0., 1.], [1., 0.], [1., 0.]]
    original = np.asarray

    def guarded(array, *args, **kwargs):
        if disk_array(array) is not None and kwargs.get('order') == 'C':
            assert array.flags.c_contiguous, 'full Fortran matrix cast into RAM'
        return original(array, *args, **kwargs)

    monkeypatch.setattr(np, 'asarray', guarded)
    result = solvers.solve_mbar(u, np.array([0, 0, 1, 1]), backend='numpy')
    np.testing.assert_allclose(result['f_k'], [0., 0.], atol=1e-12)
    np.testing.assert_allclose(result['logw'], [-np.log(4)] * 4, atol=1e-12)


def test_mapping_preserves_reserved_disk_blocks(tmp_path):
    from gareus.mbar_analysis.storage import create_matrix

    matrix = create_matrix((2000, 1000), directory=tmp_path)
    info = Path(matrix.filename).stat()
    assert info.st_blocks * 512 >= matrix.nbytes, 'mapping discarded space reservation'


def test_ladder_does_not_overwrite_caller_owned_mapped_input(tmp_path):
    original = np.arange(6.).reshape(3, 2)
    u = mapped(tmp_path, original)
    env = PepGamdEnvelope(50., -50., 50., .8, 50., -50., 50., .6)
    result = ladder.apply_ladder_boost_to_u(u, np.zeros(3), np.zeros(3),
                                           np.array([0., 1.]), env, .4, {})
    np.testing.assert_array_equal(u, original)
    assert isinstance(result, np.memmap)
    assert not np.array_equal(result, original)

"""Bounded matrix operations and process-owned temporary storage."""
from __future__ import annotations

import atexit
import errno
from functools import wraps
import mmap
import os
import tempfile
import weakref
from pathlib import Path

import numpy as np

BLOCK_ELEMENTS = 1_000_000
_paths: set[Path] = set()


def disk_array(array):
    seen = set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        if isinstance(array, np.memmap) and getattr(array, '_mmap', None) is not None:
            return array
        array = getattr(array, 'base', None)
    return None


def row_blocks(rows, columns):
    size = max(1, BLOCK_ELEMENTS // max(1, columns))
    for start in range(0, rows, size):
        yield start, min(rows, start + size)


def _remove(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        return
    _paths.discard(Path(path))


def cleanup():
    for path in tuple(_paths):
        _remove(path)


def cleanup_on_error(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        before = set(_paths)
        try:
            return function(*args, **kwargs)
        except BaseException:
            for path in set(_paths) - before:
                _remove(path)
            raise
    return wrapped


atexit.register(cleanup)


def create_matrix(shape, directory=None):
    if not all(shape):
        return np.empty(shape, dtype=np.float64)
    fd, name = tempfile.mkstemp(prefix='.gareus-matrix-', suffix='.dat', dir=directory)
    path = Path(name)
    _paths.add(path)
    try:
        size = int(shape[0]) * int(shape[1]) * 8
        try:
            os.posix_fallocate(fd, 0, size)
        except OSError as exc:
            if exc.errno in (errno.EOPNOTSUPP, errno.ENOSYS):
                raise OSError(exc.errno, 'low-memory scratch filesystem must support space reservation',
                              str(path)) from exc
            raise
        result = np.memmap(path, mode='r+', dtype=np.float64, shape=shape)
    except BaseException:
        _remove(path)
        raise
    finally:
        os.close(fd)
    weakref.finalize(result, _remove, path)
    return result


def release_pages(array, *, written=False):
    mapped = disk_array(array)
    if mapped is None or mapped.mode == 'c':
        return
    if written:
        mapped.flush()
    try:
        mapped._mmap.madvise(mmap.MADV_DONTNEED)
    except (AttributeError, OSError, ValueError):
        pass


def owned_writable_matrix(array):
    mapped = disk_array(array)
    return (mapped is not None and mapped.mode != 'c' and array.flags.writeable
            and Path(mapped.filename) in _paths)


def select_matrix(array, rows=None, columns=None):
    """Preserve disk backing through row/column fancy indexing."""
    if rows is None and columns is None:
        return array
    if disk_array(array) is None:
        out = array if rows is None else array[rows]
        return out if columns is None else np.ascontiguousarray(out[:, columns])
    n, k = array.shape
    if rows is not None:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if rows.shape != (n,):
                raise ValueError('row mask must match matrix length')
            if rows.all():
                rows = None
            else:
                rows = np.flatnonzero(rows)
    columns = None if columns is None else np.asarray(columns)
    if columns is not None and np.array_equal(columns, np.arange(k)):
        columns = None
    if rows is None and columns is None:
        return array
    shape = (n if rows is None else len(rows), k if columns is None else len(columns))
    result = create_matrix(shape, directory=Path(disk_array(array).filename).parent)
    try:
        for start, stop in row_blocks(shape[0], max(k, shape[1])):
            chunk = array[start:stop] if rows is None else array[rows[start:stop]]
            result[start:stop] = chunk if columns is None else chunk[:, columns]
            del chunk
            release_pages(result, written=True)
            release_pages(array)
    except BaseException:
        if isinstance(result, np.memmap):
            _remove(result.filename)
        raise
    return result


def finite_rows(array):
    result = np.empty(array.shape[0], dtype=bool)
    for start, stop in row_blocks(*array.shape):
        result[start:stop] = np.isfinite(array[start:stop]).all(axis=1)
        release_pages(array)
    return result


def build_matrix(shape, builder, directory=None):
    result = create_matrix(shape, directory=directory)
    try:
        for start, stop in row_blocks(*shape):
            block = builder(start, stop)
            result[start:stop] = block
            release_pages(block)
            del block
            release_pages(result, written=True)
    except BaseException:
        if isinstance(result, np.memmap):
            _remove(result.filename)
        raise
    return result


def solver_matrix(array):
    mapped = disk_array(array)
    if mapped is not None and (array.dtype != np.float64 or not array.flags.c_contiguous):
        return build_matrix(array.shape, lambda start, stop: array[start:stop],
                            directory=Path(mapped.filename).parent)
    return np.asarray(array, dtype=np.float64, order='C')


def extract_npz_matrix(npz, key, directory=None):
    """Stream a compressed NPY member to disk, then map it without decompression RAM."""
    import shutil

    fd, name = tempfile.mkstemp(prefix='.gareus-matrix-', suffix='.npy', dir=directory)
    path = Path(name)
    _paths.add(path)
    try:
        with os.fdopen(fd, 'wb') as out, npz.zip.open(key + '.npy') as source:
            shutil.copyfileobj(source, out, length=1024 * 1024)
        result = np.load(path, mmap_mode='r', allow_pickle=False)
        if result.ndim != 2 or not np.issubdtype(result.dtype, np.number):
            raise ValueError('low-memory bias matrix must be 2-D numeric')
        weakref.finalize(result, _remove, path)
        if result.dtype != np.float64 or not result.flags.c_contiguous:
            converted = build_matrix(result.shape, lambda start, stop: result[start:stop],
                                     directory=directory)
            return converted
        return result
    except BaseException:
        _remove(path)
        raise

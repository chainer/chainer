import numpy
from numpy.linalg import LinAlgError

import cupy
from cupy import core
from cupy import internal
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.cuda import cusolver


def _assertCupyArray(*arrays):
    for a in arrays:
        if not isinstance(a, cupy.core.ndarray):
            raise LinAlgError('cupy.linalg only supports cupy.core.ndarray')


def _assertRankAtLeast2(*arrays):
    for a in arrays:
        if len(a.shape) < 2:
            raise LinAlgError(
                '{}-dimensional array given. Array must be '
                'at least two-dimensional'.format(len(a.shape)))


def _assertRank2(*arrays):
    for a in arrays:
        if len(a.shape) != 2:
            raise LinAlgError(
                '{}-dimensional array given. Array must be '
                'two-dimensional'.format(len(a.shape)))


def _assertNdSquareness(*arrays):
    for a in arrays:
        if max(a.shape[-2:]) != min(a.shape[-2:]):
            raise LinAlgError('Last 2 dimensions of the array must be square')


def _tril(x, k=0):
    m, n = x.shape
    u = cupy.arange(m).reshape(m, 1)
    v = cupy.arange(n).reshape(1, n)
    mask = v - u <= k
    x *= mask
    return x


def _triu(x, k=0):
    m, n = x.shape
    u = cupy.arange(m).reshape(m, 1)
    v = cupy.arange(n).reshape(1, n)
    mask = v - u >= k
    x *= mask
    return x


def cholesky(a):
    _assertCupyArray(a)
    _assertRank2(a)
    _assertNdSquareness(a)

    ret_dtype = a.dtype.char
    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    x = a.astype(dtype, copy=True)
    n = a.shape[0]
    handle = device.get_cusolver_handle()
    devInfo = cupy.empty(1, dtype=numpy.int32)
    if x.dtype.char == 'f':
        buffersize = cusolver.spotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        cusolver.spotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    else:  # a.dtype.char == 'd'
        buffersize = cusolver.dpotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        cusolver.dpotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    status = int(devInfo[0])
    if status > 0:
        raise LinAlgError(
            'The leading minor of order {} '
            'is not positive definite'.format(status))
    elif status < 0:
        raise LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')
    _tril(x, k=0)
    return x


def qr(a, mode='default'):
    _assertCupyArray(a)
    _assertRank2(a)

    ret_dtype = a.dtype.char
    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    m, n = a.shape
    x = a.transpose().astype(dtype, copy=True)
    mn = min(m, n)
    handle = device.get_cusolver_handle()
    devInfo = cupy.empty(1, dtype=numpy.int32)
    # compute working space of geqrf and ormqr, and solve R
    if x.dtype.char == 'f':
        buffersize = cusolver.sgeqrf_bufferSize(handle, m, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        tau = cupy.empty(mn, dtype=numpy.float32)
        cusolver.sgeqrf(
            handle, m, n, x.data.ptr, m,
            tau.data.ptr, workspace.data.ptr, buffersize, devInfo.data.ptr)
    else:  # a.dtype.char == 'd'
        buffersize = cusolver.dgeqrf_bufferSize(handle, n, m, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        tau = cupy.empty(mn, dtype=numpy.float64)
        cusolver.dgeqrf(
            handle, m, n, x.data.ptr, m,
            tau.data.ptr, workspace.data.ptr, buffersize, devInfo.data.ptr)
    status = int(devInfo[0])
    if status < 0:
        raise LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')

    if mode == 'r':
        r = x[:, :mn].transpose().astype(dtype, copy=True)
        return _triu(r)

    if mode == 'raw':
        return x, tau

    if mode == 'complete':
        raise NotImplementedError(
            'Current cupy.linalg.qr does not support \'complete\' option')

    if mode == 'complete' and m > n:
        print "hoge"
        mc = m
        q = cupy.zeros((m, m), dtype=dtype)
        q[:m, :m] = cupy.identity(m, dtype=dtype)
    else:
        # XXX: This is valid in the case that m <= n only
        mc = mn
        q = cupy.zeros((n, m), dtype=dtype)
        q[:mn, :mn] = cupy.identity(mn, dtype=dtype)

    # solve Q
    # Since current CUSOLVER does not provide (s|d)orgqr,
    # we instead used the pair of (s|d)ormqr and an identity matrix.
    if x.dtype.char == 'f':
        cusolver.sormqr(
            handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_OP_T,
            m, n, mn, x.data.ptr, m, tau.data.ptr, q.data.ptr, m,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    else:
        cusolver.dormqr(
            handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_OP_T,
            m, n, mn, x.data.ptr, m, tau.data.ptr, q.data.ptr, m,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    q = q[:mc].astype(dtype, copy=True)
    r = x[:, :mc].transpose().astype(dtype, copy=True)
    return q, _triu(r)


# TODO(okuta): Implement svd

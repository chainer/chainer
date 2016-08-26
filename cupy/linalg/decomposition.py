import numpy
from numpy.linalg import LinAlgError

import cupy
from cupy.cuda import cublas
from cupy.cuda import cusolver
from cupy.cuda import device


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
    # TODO(Saito): Current implementation only accepts two-dimensional arrays
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


def qr(a, mode='reduced'):
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
        if ret_dtype == 'f':
            # In the case of raw mode, The type of x and tau in numpy
            # is float32 because numpy always computes qr() as float64.
            # Considering the meaning of 'raw' we consider this conversion
            # would be inappropriate, however, in this time we convert the
            # both variables into float64 for compatibility.
            return x.astype(numpy.float64), tau.astype(numpy.float64)
        return x, tau

    if mode == 'complete':
        raise NotImplementedError(
            'Current cupy.linalg.qr does not support \'complete\' option')

    if m <= n:
        mc = m
        q = cupy.zeros((n, m), dtype=dtype)
        q[:mn, :mn] = cupy.identity(mn, dtype=dtype)
    else:
        raise NotImplementedError()

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


def svd(a, full_matrices=True, compute_uv=True):
    # TODO(Saito): Current implementation only accepts two-dimensional arrays
    _assertCupyArray(a)
    _assertRank2(a)

    if not (full_matrices and compute_uv):
        raise NotImplementedError(
            'Current CUSOLVER only supports SVD generating full marices')

    ret_dtype = a.dtype.char
    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    # Remark 1: gesvd only supports m >= n (WHAT?)
    # Remark 2: gesvd only supports jobu = 'A' and jobvt = 'A'
    #           and returns matrix U and V^H
    n, m = a.shape
    if m >= n:
        x = a.astype(dtype, copy=True)
        trans_flag = False
    else:
        m, n = a.shape
        x = a.transpose().astype(dtype, copy=True)
        trans_flag = True
    mn = min(m, n)

    u = cupy.empty((m, m), dtype=dtype)
    s = cupy.empty(mn, dtype=dtype)
    vt = cupy.empty((n, n), dtype=dtype)
    handle = device.get_cusolver_handle()
    devInfo = cupy.empty(1, dtype=numpy.int32)
    jobu, jobvt = ord('A'), ord('A')
    if x.dtype.char == 'f':
        buffersize = cusolver.sgesvd_bufferSize(handle, m, n)
        workspace = cupy.empty(buffersize, dtype=dtype)
        cusolver.sgesvd(
            handle, jobu, jobvt, m, n, x.data.ptr, m,
            s.data.ptr, u.data.ptr, m, vt.data.ptr, n,
            workspace.data.ptr, buffersize, 0, devInfo.data.ptr)
    else:
        buffersize = cusolver.dgesvd_bufferSize(handle, m, n)
        workspace = cupy.empty(buffersize, dtype=dtype)
        cusolver.dgesvd(
            handle, jobu, jobvt, m, n, x.data.ptr, m,
            s.data.ptr, u.data.ptr, m, vt.data.ptr, n,
            workspace.data.ptr, buffersize, 0, devInfo.data.ptr)

    status = int(devInfo[0])
    if status > 0:
        raise LinAlgError(
            'SVD computation does not converge')
    elif status < 0:
        raise LinAlgError(
            'Parameter error (maybe caused by a bug in cupy.linalg?)')

    # Note that the returned array may need to be transporsed
    # depending on the structure of an input
    if trans_flag:
        return u.transpose(), s, vt.transpose()
    else:
        return vt, s, u

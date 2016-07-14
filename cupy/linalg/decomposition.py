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


def _assertNdSquareness(*arrays):
    for a in arrays:
        if max(a.shape[-2:]) != min(a.shape[-2:]):
            raise LinAlgError('Last 2 dimensions of the array must be square')


def _tril(x, k=0):
    n, _ = x.shape
    ind = cupy.arange(n)
    u = ind.reshape(1, n)
    v = ind.reshape(n, 1)
    if k > 0:
        mask = (u >= v)
    elif k < 0:
        mask = (u <= v)
    else:
        mask = (u == v)
    x *= mask
    return x


def cholesky(a):
    _assertCupyArray(a)
    _assertNdSquareness(a)
    if a.ndim != 2:
        raise LinAlgError(
            'The current cholesky() supports'
            'two-dimensional array only')

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
    if a.dtype.char == 'f':
        buffersize = cusolver.spotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        cusolver.spotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    else:
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
    _tril(x, k=-1)
    return x

# TODO(okuta): Implement qr

# TODO(okuta): Implement svd

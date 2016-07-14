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
            raise LinAlgError('%d-dimensional array given. Array must be '
                'at least two-dimensional' % len(a.shape))


def _assertNdSquareness(*arrays):
    for a in arrays:
        if max(a.shape[-2:]) != min(a.shape[-2:]):
            raise LinAlgError('Last 2 dimensions of the array must be square')

def _tril(x, k=0):
    n, _ = x.shape
    u = cupy.arange(n).reshape(1, n)
    v = cupy.arange(n).reshape(n, 1)
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
    _assertRankAtLeast2(a)
    _assertNdSquareness(a)
    ret_dtype = a.dtype.char
    # Cast to float32 or float64
    if ret_dtype == 'f' or ret_dtype == 'd':
        dtype = ret_dtype
    else:
        dtype = numpy.find_common_type((ret_dtype, 'f'), ()).char

    x = a.astype(dtype, copy=True)
    handle = device.get_cusolver_handle()
    n = a.shape[0]
    if a.dtype.char == 'f':
        buffersize = cusolver.spotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float32)
        devInfo = cupy.empty(1, dtype=numpy.int32)
        cusolver.spotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    else:
        buffersize = cusolver.dpotrf_bufferSize(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
        workspace = cupy.empty(buffersize, dtype=numpy.float64)
        devInfo = cupy.empty(1, dtype=numpy.int32)
        cusolver.dpotrf(
            handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
            workspace.data.ptr, buffersize, devInfo.data.ptr)
    _tril(x, k=-1)
    return x

# TODO(okuta): Implement qr


# TODO(okuta): Implement svd

import os

try:
    import cupy
except Exception:
    cupy = None

import chainerx


_cuda_limit = None


def get_cuda_limit():
    global _cuda_limit
    if _cuda_limit is not None:
        return _cuda_limit
    if os.getenv('CHAINERX_TEST_CUDA_DEVICE_LIMIT') is None:
        try:
            backend = chainerx.get_global_default_context().get_backend('cuda')
            _cuda_limit = backend.get_device_count()
        except chainerx.BackendError:
            _cuda_limit = 0
    else:
        _cuda_limit = int(os.getenv('CHAINERX_TEST_CUDA_DEVICE_LIMIT'))
        if _cuda_limit < 0:
            raise chainerx.ChainerxError(
                'CHAINERX_TEST_DUDA_DEVICE_LIMIT must be non-negative '
                'integer: {}'.format(_cuda_limit))
    return _cuda_limit


def get_current_device():
    # Returns the current CUDA device.
    # Returns None if cupy is not installed.
    # TODO(niboshi): Better remove dependency to cupy
    if cupy is None:
        return None
    return cupy.cuda.runtime.getDevice()

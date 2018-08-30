import os

import xchainer


_cuda_limit = None


def get_cuda_limit():
    global _cuda_limit
    if _cuda_limit is not None:
        return _cuda_limit
    if os.getenv('XCHAINER_TEST_CUDA_DEVICE_LIMIT') is None:
        try:
            backend = xchainer.get_global_default_context().get_backend('cuda')
            _cuda_limit = backend.get_device_count()
        except xchainer.BackendError:
            _cuda_limit = 0
    else:
        _cuda_limit = int(os.getenv('XCHAINER_TEST_CUDA_DEVICE_LIMIT'))
        if _cuda_limit < 0:
            raise xchainer.XchainerError('XCHAINER_TEST_DUDA_DEVICE_LIMIT must be non-negative integer: {}'.format(_cuda_limit))
    return _cuda_limit

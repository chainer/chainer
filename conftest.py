import os

import pytest

import xchainer

def pytest_configure(config):
    _register_cuda_marker(config)


def pytest_runtest_setup(item):
    _setup_cuda_marker(item)


def _register_cuda_marker(config):
    config.addinivalue_line("markers", "cuda(num=1): mark tests needing the specified number of NVIDIA GPUs.")


def _setup_cuda_marker(item):
    """Pytest marker to indicate number of NVIDIA GPUs required to run the test.

    Tests can be annotated with this decorator (e.g., ``@pytest.mark.cuda``) to
    declare that one NVIDIA GPU is required to run.

    Tests can also be annotated as ``@pytest.mark.cuda(2)`` to declare number of
    NVIDIA GPUs required to run. When running tests, if ``XCHAINER_TEST_CUDA_DEVICE_LIMIT``
    environment variable is set to value greater than or equals to 0, test cases
    that require GPUs more than the limit will be skipped.
    """

    cuda_marker = item.get_marker('cuda')
    if cuda_marker is not None:
        required_num = cuda_marker.args[0] if cuda_marker.args else 1
        if _get_cuda_limit() < required_num:
            pytest.skip('{} NVIDIA GPUs required'.format(required_num))


_cuda_limit = None

def _get_cuda_limit():
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
    return _cuda_limit

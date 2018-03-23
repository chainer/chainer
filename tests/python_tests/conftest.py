import pytest

import xchainer

from tests import cuda_utils


_dtypes = [
    xchainer.bool,
    xchainer.int8,
    xchainer.int16,
    xchainer.int32,
    xchainer.int64,
    xchainer.uint8,
    xchainer.float32,
    xchainer.float64,
]


_float_dtypes = [
    xchainer.float32,
    xchainer.float64,
]


_signed_dtypes = [
    xchainer.int8,
    xchainer.int16,
    xchainer.int32,
    xchainer.int64,
    xchainer.float32,
    xchainer.float64,
]


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.function, 'parametrize_device'):
        device_names, = metafunc.function.parametrize_device.args
        metafunc.parametrize('device', device_names, indirect=True)


# TODO(sonots): xchainer python main program should create default backend rather than creating at here
@pytest.fixture(scope='session', autouse=True)
def scope_session():
    device = xchainer.get_global_default_context().get_device('native', 0)
    with xchainer.device_scope(device):
        yield


def _get_required_cuda_devices_from_device_name(device_name):
    s = device_name.split(':')
    assert len(s) == 2
    if s[0] != 'cuda':
        return 0
    return int(s[1]) + 1


@pytest.fixture
def device(request):
    device_name = request.param
    cuda_device_count = _get_required_cuda_devices_from_device_name(device_name)
    if cuda_device_count > cuda_utils.get_cuda_limit():
        pytest.skip()
    device = xchainer.get_device(device_name)
    device_scope = xchainer.device_scope(device)

    def finalize():
        device_scope.__exit__()

    request.addfinalizer(finalize)
    device_scope.__enter__()
    return device


@pytest.fixture(params=_dtypes)
def dtype(request):
    return request.param


@pytest.fixture(params=_float_dtypes)
def float_dtype(request):
    return request.param


@pytest.fixture(params=_signed_dtypes)
def signed_dtype(request):
    return request.param

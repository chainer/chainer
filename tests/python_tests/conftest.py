import pytest

import xchainer


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


# TODO(sonots): xchainer python main program should create default backend rather than creating at here
@pytest.fixture(scope='session', autouse=True)
def scope_session():
    device = xchainer.get_global_default_context().get_device('native', 0)
    with xchainer.device_scope(device):
        yield


@pytest.fixture(params=_dtypes)
def dtype(request):
    return request.param


@pytest.fixture(params=_float_dtypes)
def float_dtype(request):
    return request.param


@pytest.fixture(params=_signed_dtypes)
def signed_dtype(request):
    return request.param

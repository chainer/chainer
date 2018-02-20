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
    backend = xchainer.NativeBackend()
    device_id = xchainer.DeviceId(backend)
    with xchainer.device_id_scope(device_id):
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

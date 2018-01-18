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


@pytest.fixture(params=_dtypes)
def dtype(request):
    return request.param


@pytest.fixture(params=_float_dtypes)
def float_dtype(request):
    return request.param


@pytest.fixture(params=_signed_dtypes)
def signed_dtype(request):
    return request.param

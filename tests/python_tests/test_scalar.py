import pytest

import xchainer


def _check_cast_scalar_equals_data(scalar, data):
    assert bool(scalar) == bool(data)
    assert int(scalar) == int(data)
    assert float(scalar) == float(data)


_scalars_data = [
    {'data': -2},
    {'data': 1},
    {'data': -1.5},
    {'data': 2.3},
    {'data': True},
    {'data': False},
]


@pytest.fixture(params=_scalars_data)
def scalar_data(request):
    return request.param


@pytest.fixture
def scalar_init_inputs(request, scalar_data):
    return scalar_data['data']


def test_cast(scalar_init_inputs):
    data = scalar_init_inputs
    scalar = xchainer.Scalar(data)

    _check_cast_scalar_equals_data(scalar, data)
    _check_cast_scalar_equals_data(+scalar, +data)
    if isinstance(data, bool):
        with pytest.raises(xchainer.DtypeError):
            -scalar  # should not be able to negate bool
    else:
        _check_cast_scalar_equals_data(-scalar, -data)


def test_dtype(scalar_init_inputs):
    data = scalar_init_inputs
    scalar = xchainer.Scalar(data)

    if isinstance(data, bool):
        assert scalar.dtype == xchainer.bool
    elif isinstance(data, int):
        assert scalar.dtype == xchainer.int64
    elif isinstance(data, float):
        assert scalar.dtype == xchainer.float64
    else:
        assert False


def test_repr(scalar_init_inputs):
    data = scalar_init_inputs
    scalar = xchainer.Scalar(data)

    assert repr(scalar) == repr(data)
    assert str(scalar) == str(data)


def test_init_invalid():
    with pytest.raises(TypeError):
        xchainer.Scalar("1")  # string, which is not a numeric

import pytest

import xchainer


@pytest.fixture
def inputs(request, scalar_data):
    return scalar_data['data']


def _assert_casts_equal(scalar, data):
    assert bool(scalar) == bool(data)
    assert int(scalar) == int(data)
    assert float(scalar) == float(data)


def test_cast(inputs):
    data = inputs
    scalar = xchainer.Scalar(data)

    _assert_casts_equal(scalar, data)
    _assert_casts_equal(+scalar, +data)
    if isinstance(data, bool):
        with pytest.raises(xchainer.DtypeError):
            -scalar  # should not be able to negate bool
    else:
        _assert_casts_equal(-scalar, -data)


def test_dtype(inputs):
    data = inputs
    scalar = xchainer.Scalar(data)

    if isinstance(data, bool):
        assert scalar.dtype == xchainer.bool
    elif isinstance(data, int):
        assert scalar.dtype == xchainer.int64
    elif isinstance(data, float):
        assert scalar.dtype == xchainer.float64
    else:
        assert False


def test_repr(inputs):
    data = inputs
    scalar = xchainer.Scalar(data)

    assert repr(scalar) == repr(data)
    assert str(scalar) == str(data)


def test_init_invalid():
    with pytest.raises(TypeError):
        xchainer.Scalar("1")  # string, which is not a numeric

import pytest

import xchainer


@pytest.fixture(params=[-2, 1, -1.5, 2.3, True, False])
def scalar_data(request):
    return request.param


def assert_casts(scalar, scalar_data):
    assert bool(scalar) == bool(scalar_data)
    assert int(scalar) == int(scalar_data)
    assert float(scalar) == float(scalar_data)


def check_cast(scalar_data):
    scalar = xchainer.Scalar(scalar_data)
    assert_casts(scalar, scalar_data)


# operator+()
def check_pos_cast(scalar_data):
    scalar = xchainer.Scalar(scalar_data)
    assert_casts(+scalar, +scalar_data)


# operator-()
def check_neg_cast(scalar_data):
    scalar = xchainer.Scalar(scalar_data)

    if isinstance(scalar_data, bool):
        with pytest.raises(xchainer.DtypeError):
            -scalar  # Should not be able to negate bool
    else:
        assert_casts(-scalar, -scalar_data)


def test_cast(scalar_data):
    check_cast(scalar_data)
    check_pos_cast(scalar_data)
    check_neg_cast(scalar_data)


def test_repr(scalar_data):
    assert repr(xchainer.Scalar(scalar_data)) == repr(scalar_data)
    assert str(xchainer.Scalar(scalar_data)) == str(scalar_data)


def test_dtype(scalar_data):
    scalar = xchainer.Scalar(scalar_data)
    if isinstance(scalar_data, bool):
        assert scalar.dtype == xchainer.bool
    elif isinstance(scalar_data, int):
        assert scalar.dtype == xchainer.int64
    elif isinstance(scalar_data, float):
        assert scalar.dtype == xchainer.float64
    else:
        assert False


def test_init_invalid():
    with pytest.raises(TypeError):
        xchainer.Scalar("1")

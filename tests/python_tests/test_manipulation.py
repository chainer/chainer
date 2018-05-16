import math

import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


_shapes = [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
]


@pytest.fixture(params=_shapes)
def shape(request):
    return request.param


@pytest.mark.parametrize('value', [
    0, 1, -1, 0.1, 0.9, -0.1, -0.9, 1.1, -1.1, 1.9, -1.9, True, False, float('inf'), -float('inf'), float('nan'), -0.0
])
@pytest.mark.parametrize('shape', [
    (), (1,), (1, 1, 1)
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_asscalar(device, value, shape, dtype):
    np_dtype = numpy.dtype(dtype)
    try:
        np_value = np_dtype.type(value)
    except (ValueError, OverflowError):
        return

    a_np = numpy.asarray([np_value], dtype).reshape(shape)
    a_xc = xchainer.array(a_np)

    def should_cast_succeed(typ):
        try:
            typ(np_value)
            return True
        except (ValueError, OverflowError):
            return False

    # Cast to float
    if should_cast_succeed(float):
        assert type(float(a_xc)) is float
        if math.isnan(float(a_np)):
            assert math.isnan(float(a_xc))
        else:
            assert float(a_np) == float(a_xc)
    # Cast to int
    if should_cast_succeed(int):
        assert type(int(a_xc)) is int
        assert int(a_np) == int(a_xc)
    # Cast to bool
    if should_cast_succeed(bool):
        assert type(bool(a_xc)) is bool
        assert bool(a_np) == bool(a_xc)

    # xchainer.asscalar
    assert isinstance(xchainer.asscalar(a_xc), type(numpy.asscalar(a_np)))
    if math.isnan(numpy.asscalar(a_np)):
        assert math.isnan(xchainer.asscalar(a_xc))
    else:
        assert xchainer.asscalar(a_xc) == numpy.asscalar(a_np)


@pytest.mark.parametrize('shape', [
    (0,), (1, 0), (2,), (1, 2), (2, 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_invalid_asscalar(device, shape):
    dtype = xchainer.float32

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        xchainer.asscalar(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        float(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        int(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        bool(a)


@xchainer.testing.numpy_xchainer_array_equal()
def test_transpose(xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    return array.transpose()


@xchainer.testing.numpy_xchainer_array_equal()
def test_T(xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    return array.T


@xchainer.testing.numpy_xchainer_array_equal()
def test_module_transpose(xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    return xp.transpose(array)

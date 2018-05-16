import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


@pytest.mark.parametrize('a_object,b_object', [
    ([], []),
    ([0], [0]),
    ([0], [-0]),
    ([0], [1]),
    ([0.2], [0.2]),
    ([0.2], [-0.3]),
    ([True], [True]),
    ([True], [False]),
    ([0, 1, 2], [0, 1, 2]),
    ([1, 1, 2], [0, 1, 2]),
    ([0, 1, 2], [1, 2, 3]),
    ([0., numpy.nan], [0., 1.]),
    ([0., numpy.nan], [0., numpy.nan]),
    ([0., numpy.inf], [0., 1.]),
    ([0., -numpy.inf], [0., 1.]),
    ([numpy.inf, 1.], [numpy.inf, 1.]),
    ([-numpy.inf, 1.], [-numpy.inf, 1.]),
    ([numpy.inf, 1.], [-numpy.inf, 1.]),
    ([numpy.inf, 1.], [-numpy.inf, numpy.nan]),
    ([[0, 1], [2, 3]], [[0, 1], [2, 3]]),
    ([[0, 1], [2, 3]], [[0, 1], [2, -2]]),
    ([[0, 1], [2, 3]], [[1, 2], [3, 4]]),
    # broadcast
    (0, [0]),
    (1, [0]),
    ([], [0]),
    ([0], [[0, 1, 2], [3, 4, 5]]),
    ([[0], [1]], [0, 1, 2]),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eq(device, a_object, b_object, dtype):
    try:
        a_np = numpy.array(a_object, dtype)
        b_np = numpy.array(b_object, dtype)
    except (ValueError, OverflowError):
        # Skip if creating an ndarray while casting the data to the parameterized dtype fails.
        # E.g. [numpy.inf] to numpy.int32.
        return

    a_xc = xchainer.array(a_np)
    b_xc = xchainer.array(b_np)

    xchainer.testing.assert_array_equal_ex(a_xc == b_xc, a_np == b_np)
    xchainer.testing.assert_array_equal_ex(b_xc == a_xc, b_np == a_np)
    xchainer.testing.assert_array_equal_ex(xchainer.equal(a_xc, b_xc), numpy.equal(a_np, b_np))
    xchainer.testing.assert_array_equal_ex(xchainer.equal(b_xc, a_xc), numpy.equal(b_np, a_np))


@pytest.mark.parametrize('a_shape,b_shape', [
    ((2,), (3,)),
    ((2,), (2, 3)),
    ((1, 2, 3), (1, 2, 3, 4)),
])
def test_invalid_eq(a_shape, b_shape):
    def check(x, y):
        with pytest.raises(xchainer.DimensionError):
            x == y

        with pytest.raises(xchainer.DimensionError):
            xchainer.equal(x, y)

    a = array_utils.create_dummy_ndarray(xchainer, a_shape, 'float32')
    b = array_utils.create_dummy_ndarray(xchainer, b_shape, 'float32')
    check(a, b)
    check(b, a)

import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils


# Skip if creating an ndarray while casting the data to the parameterized dtype fails.
# E.g. [numpy.inf] to numpy.int32.
def _to_array_safe(xp, a_object, dtype):
    try:
        return xp.array(a_object, dtype)
    except (ValueError, OverflowError):
        return None


# Ignore warnings from numpy for NaN comparisons.
@pytest.mark.filterwarnings('ignore:invalid value encountered in ')
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
@pytest.mark.parametrize('cmp_op, chx_cmp, np_cmp', [
    (lambda a, b: a == b, chainerx.equal, numpy.equal),
    (lambda a, b: a != b, chainerx.not_equal, numpy.not_equal),
    (lambda a, b: a > b, chainerx.greater, numpy.greater),
    (lambda a, b: a >= b, chainerx.greater_equal, numpy.greater_equal),
    (lambda a, b: a < b, chainerx.less, numpy.less),
    (lambda a, b: a <= b, chainerx.less_equal, numpy.less_equal),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cmp(device, cmp_op, chx_cmp, np_cmp, a_object, b_object, dtype):
    a_np = _to_array_safe(numpy, a_object, dtype)
    b_np = _to_array_safe(numpy, b_object, dtype)
    if a_np is None or b_np is None:
        return

    a_chx = chainerx.array(a_np)
    b_chx = chainerx.array(b_np)

    chainerx.testing.assert_array_equal_ex(cmp_op(a_chx, b_chx), cmp_op(a_np, b_np))
    chainerx.testing.assert_array_equal_ex(cmp_op(b_chx, a_chx), cmp_op(b_np, a_np))
    chainerx.testing.assert_array_equal_ex(chx_cmp(a_chx, b_chx), np_cmp(a_np, b_np))
    chainerx.testing.assert_array_equal_ex(chx_cmp(b_chx, a_chx), np_cmp(b_np, a_np))


@pytest.mark.parametrize('a_shape,b_shape', [
    ((2,), (3,)),
    ((2,), (2, 3)),
    ((1, 2, 3), (1, 2, 3, 4)),
])
@pytest.mark.parametrize('cmp_op, chx_cmp', [
    (lambda a, b: a == b, chainerx.equal),
    (lambda a, b: a != b, chainerx.not_equal),
    (lambda a, b: a > b, chainerx.greater),
    (lambda a, b: a >= b, chainerx.greater_equal),
    (lambda a, b: a < b, chainerx.less),
    (lambda a, b: a < b, chainerx.less_equal),
])
def test_cmp_invalid(cmp_op, chx_cmp, a_shape, b_shape):
    def check(x, y):
        with pytest.raises(chainerx.DimensionError):
            cmp_op(x, y)

        with pytest.raises(chainerx.DimensionError):
            chx_cmp(x, y)

    a = array_utils.create_dummy_ndarray(chainerx, a_shape, 'float32')
    b = array_utils.create_dummy_ndarray(chainerx, b_shape, 'float32')
    check(a, b)
    check(b, a)


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('a_object', [
    ([]),
    ([0]),
    ([1]),
    ([0.2]),
    ([-0.3]),
    ([True]),
    ([False]),
    ([0, 1, 2]),
    ([0., numpy.nan]),
    ([numpy.nan, numpy.inf]),
    ([-numpy.inf, numpy.nan]),
    ([[0, 1], [2, 0]]),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_logical_not(xp, device, a_object, dtype):
    a = _to_array_safe(xp, a_object, dtype)
    if a is None:
        return chainerx.testing.ignore()
    return xp.logical_not(a)

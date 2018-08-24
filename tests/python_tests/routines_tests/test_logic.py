import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


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
@pytest.mark.parametrize('cmp_op, xc_cmp, np_cmp', [
    (lambda a, b: a == b, xchainer.equal, numpy.equal),
    (lambda a, b: a != b, xchainer.not_equal, numpy.not_equal),
    (lambda a, b: a > b, xchainer.greater, numpy.greater),
    (lambda a, b: a >= b, xchainer.greater_equal, numpy.greater_equal),
    (lambda a, b: a < b, xchainer.less, numpy.less),
    (lambda a, b: a <= b, xchainer.less_equal, numpy.less_equal),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cmp(device, cmp_op, xc_cmp, np_cmp, a_object, b_object, dtype):
    a_np = _to_array_safe(numpy, a_object, dtype)
    b_np = _to_array_safe(numpy, b_object, dtype)
    if a_np is None or b_np is None:
        return

    a_xc = xchainer.array(a_np)
    b_xc = xchainer.array(b_np)

    xchainer.testing.assert_array_equal_ex(cmp_op(a_xc, b_xc), cmp_op(a_np, b_np))
    xchainer.testing.assert_array_equal_ex(cmp_op(b_xc, a_xc), cmp_op(b_np, a_np))
    xchainer.testing.assert_array_equal_ex(xc_cmp(a_xc, b_xc), np_cmp(a_np, b_np))
    xchainer.testing.assert_array_equal_ex(xc_cmp(b_xc, a_xc), np_cmp(b_np, a_np))


@pytest.mark.parametrize('a_shape,b_shape', [
    ((2,), (3,)),
    ((2,), (2, 3)),
    ((1, 2, 3), (1, 2, 3, 4)),
])
@pytest.mark.parametrize('cmp_op, xc_cmp', [
    (lambda a, b: a == b, xchainer.equal),
    (lambda a, b: a > b, xchainer.greater),
])
def test_cmp_invalid(cmp_op, xc_cmp, a_shape, b_shape):
    def check(x, y):
        with pytest.raises(xchainer.DimensionError):
            cmp_op(x, y)

        with pytest.raises(xchainer.DimensionError):
            xc_cmp(x, y)

    a = array_utils.create_dummy_ndarray(xchainer, a_shape, 'float32')
    b = array_utils.create_dummy_ndarray(xchainer, b_shape, 'float32')
    check(a, b)
    check(b, a)


@xchainer.testing.numpy_xchainer_array_equal()
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
        return xchainer.testing.ignore()
    return xp.logical_not(a)

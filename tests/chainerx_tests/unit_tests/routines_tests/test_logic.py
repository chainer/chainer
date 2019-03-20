import unittest

import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import op_utils


# Skip if creating an ndarray while casting the data to the parameterized dtype
# fails.
# E.g. [numpy.inf] to numpy.int32.
def _to_numpy_array_or_skip(a_object, dtype):
    try:
        return numpy.array(a_object, dtype)
    except (ValueError, OverflowError):
        raise unittest.SkipTest('Invalid input combination')


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('a_object,b_object', [
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
@chainer.testing.parameterize_pytest('cmp_op,module_func', [
    (lambda a, b: a == b, 'equal'),
    (lambda a, b: a != b, 'not_equal'),
    (lambda a, b: a > b, 'greater'),
    (lambda a, b: a >= b, 'greater_equal'),
    (lambda a, b: a < b, 'less'),
    (lambda a, b: a <= b, 'less_equal'),
])
# Ignore warnings from numpy for NaN comparisons.
@pytest.mark.filterwarnings('ignore:invalid value encountered in ')
class TestCmp(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def setup(self, dtype):
        a_object = self.a_object
        b_object = self.b_object
        self.a_np = _to_numpy_array_or_skip(a_object, dtype)
        self.b_np = _to_numpy_array_or_skip(b_object, dtype)

    def generate_inputs(self):
        a = self.a_np
        b = self.b_np
        return a, b

    def forward_xp(self, inputs, xp):
        a, b = inputs
        cmp_op = self.cmp_op
        module_func = getattr(xp, self.module_func)

        y1 = cmp_op(a, b)
        y2 = cmp_op(b, a)
        y3 = module_func(a, b)
        y4 = module_func(b, a)
        return y1, y2, y3, y4


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


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('a_object', [
    [],
    [0],
    [1],
    [0.2],
    [-0.3],
    [True],
    [False],
    [0, 1, 2],
    [0., numpy.nan],
    [numpy.nan, numpy.inf],
    [-numpy.inf, numpy.nan],
    [[0, 1], [2, 0]],
])
class TestLogicalNot(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True

    def setup(self, dtype):
        a_object = self.a_object
        self.a_np = _to_numpy_array_or_skip(a_object, dtype)

    def generate_inputs(self):
        a = self.a_np
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        b = xp.logical_not(a)
        return b,

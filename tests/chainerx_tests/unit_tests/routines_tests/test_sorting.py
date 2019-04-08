import unittest

import chainer
import numpy

import chainerx.testing

from chainerx_tests import op_utils


_min_max_single_axis_params = [
    # input, axis
    # valid params
    (numpy.asarray(0), None),
    (numpy.asarray(-1), None),
    (numpy.asarray(float('inf')), None),
    (numpy.asarray(float('nan')), None),
    (numpy.asarray(-float('inf')), None),
    (numpy.asarray([4, 1, 4, 1]), None),
    (numpy.asarray([4, 1, 4, 1]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]), 0),
    (numpy.asarray([[4, 4, 1, 1], [4, 1, 4, 1]]).T, 1),
    (numpy.asarray([-2, -3, -1]), 0),
    (numpy.asarray([-0.0, +0.0, +0.0, -0.0]), None),
    (numpy.asarray([[True, True, False, False],
                    [True, False, True, False]]), 0),
    (numpy.ones((2, 0, 3)), 2),
    (numpy.ones((2, 3)), 1),
    (numpy.ones((2, 3)), -2),
    # invalid params
    (numpy.ones((0,)), None),
    (numpy.ones((2, 0, 3)), 1),
    (numpy.ones((2, 0, 3)), None),
    (numpy.ones((2, 3)), 2),
    (numpy.ones((2, 3)), -3),
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('input,axis', _min_max_single_axis_params)
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestArgmax(op_utils.NumpyOpTest):

    skip_backward_test = True
    skip_double_backward_test = True
    forward_accept_errors = (ValueError, chainerx.DimensionError)

    def setup(self, dtype):
        try:
            a_np = self.input.astype(dtype)
        except (ValueError, OverflowError):
            raise unittest.SkipTest('invalid combination of data and dtype')

        self.a_np = a_np

    def generate_inputs(self):
        return self.a_np,

    def forward_xp(self, inputs, xp):
        a, = inputs
        axis = self.axis
        if self.is_module:
            b = xp.argmax(a, axis)
        else:
            b = a.argmax(axis)
        return b,

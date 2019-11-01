import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing


@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']
    })
)
class TestScale(testing.FunctionTestCase):

    def setUp(self):
        self.axis = 1

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        return x1, x2

    def forward_expected(self, inputs):
        x1, x2 = inputs
        y_expected = numpy.copy(x1)
        for i, j, k in numpy.ndindex(y_expected.shape):
            y_expected[i, j, k] *= x2[j]
        return y_expected,

    def forward(self, inputs, device):
        x1, x2 = inputs
        y = functions.scale(x1, x2, self.axis)
        return y,


class TestScaleInvalidShape(unittest.TestCase):

    def test_scale_invalid_shape(self):
        x1 = chainer.Variable(numpy.zeros((3, 2, 3), numpy.float32))
        x2 = chainer.Variable(numpy.zeros((2), numpy.float32))
        axis = 0
        with chainer.using_config('debug', True):
            with self.assertRaises(AssertionError):
                functions.scale(x1, x2, axis)


testing.run_module(__name__, __file__)

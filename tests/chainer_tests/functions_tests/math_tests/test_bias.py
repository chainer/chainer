import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing
from chainer import utils


@testing.fix_random()
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
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestBias(testing.FunctionTestCase):

    skip_double_backward_test = True

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, (3, 2, 3)).astype(numpy.float32)
        x2 = numpy.random.uniform(-1, 1, (2)).astype(numpy.float32)
        return x1, x2

    def forward(self, inputs, device):
        x1, x2 = inputs
        axis = 1
        return functions.bias(x1, x2, axis),

    def forward_expected(self, inputs):
        x1, x2 = inputs
        expected = numpy.copy(x1)
        for i, j, k in numpy.ndindex(expected.shape):
            expected[i, j, k] += x2[j]
        expected = utils.force_array(expected)
        return expected,


class TestBiasInvalidShape(unittest.TestCase):

    def test_bias_invalid_shape(self):
        x1 = chainer.Variable(numpy.zeros((3, 2, 3), numpy.float32))
        x2 = chainer.Variable(numpy.zeros((2), numpy.float32))
        axis = 0
        with chainer.using_config('debug', True):
            with self.assertRaises(AssertionError):
                functions.bias(x1, x2, axis)


testing.run_module(__name__, __file__)

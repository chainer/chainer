import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'in_shape': [(4, 3, 2)],
    'out_shape': [(2, 2, 6), (2, -1, 6)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ]
)
class TestReshape(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y_expect = x.reshape(self.out_shape)
        return y_expect,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.reshape(x, self.out_shape)
        return y,


class TestReshapeSkip(unittest.TestCase):

    shape = (2, 3)

    def setUp(self):
        self.data = numpy.random.uniform(0, 1, self.shape)

    def test_ndarray(self):
        ret = functions.reshape(self.data, self.shape)
        self.assertIs(self.data, ret.data)

    def test_variable(self):
        x = chainer.Variable(self.data)
        ret = functions.reshape(x, self.shape)
        self.assertIs(x, ret)


testing.run_module(__name__, __file__)

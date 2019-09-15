import unittest

import numpy

import chainer
import chainer.functions as F
from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
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
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']
    })
)
@testing.with_requires('scipy')
class TestZeta(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-4}
        self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-4}
        self.check_double_backward_options = {'atol': 1e-4, 'rtol': 1e-4}

    def generate_inputs(self):
        x = numpy.array(numpy.random.uniform(5, 10))
        q = numpy.array(numpy.random.uniform(1, 10))
        return x, q

    def forward_expected(self, inputs):
        x, q = inputs
        import scipy
        y_expect = scipy.special.zeta(x, q)
        return numpy.array(y_expect),

    def forward(self, inputs, device):
        x, q = inputs
        y = F.zeta(x, q)
        return y,


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
@testing.without_requires('scipy')
class TestZetaExceptions(unittest.TestCase):
    def setUp(self):
        self.x = \
            numpy.array(numpy.random.uniform(1, 10))
        self.q = numpy.array(numpy.random.uniform(5, 10))
        self.func = F.zeta

    def check_forward(self, q_data, x_data):
        x = chainer.Variable(x_data)
        q = chainer.Variable(q_data)
        with self.assertRaises(ImportError):
            self.func(q, x)

    def test_zeta_forward_cpu(self):
        self.check_forward(self.q, self.x)


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
import chainer.functions as F
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(), (3, 2)],
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
class TestPolyGamma(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-4}
        self.check_backward_options = {'eps': 1e-3, 'atol': 5e-2, 'rtol': 1e-3}
        self.check_double_backward_options = {'eps': 1e-3, 'atol': 5e-2,
                                              'rtol': 1e-3}

    def generate_inputs(self):
        n = numpy.random.randint(3, size=self.shape).astype(numpy.int32)
        x = numpy.random.uniform(1., 10., self.shape).astype(self.dtype)
        return n, x

    def forward_expected(self, inputs):
        n, x = inputs
        import scipy
        y_expect = scipy.special.polygamma(n, x)
        return y_expect.astype(self.dtype),

    def forward(self, inputs, device):
        n, x = inputs
        y = F.polygamma(n, x)
        return y,


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
@testing.without_requires('scipy')
class TestPolyGammaExceptions(unittest.TestCase):
    def setUp(self):
        self.x = \
            numpy.random.uniform(1., 10., self.shape).astype(self.dtype)
        self.n = numpy.random.randint(3, size=self.shape).astype(numpy.int32)
        self.func = F.polygamma

    def check_forward(self, n_data, x_data):
        x = chainer.Variable(x_data)
        n = chainer.Variable(n_data)
        with self.assertRaises(ImportError):
            self.func(n, x)

    def test_polygamma_forward_cpu(self):
        self.check_forward(self.n, self.x)


testing.run_module(__name__, __file__)

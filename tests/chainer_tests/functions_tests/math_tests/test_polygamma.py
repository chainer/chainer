import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(), (3, 2)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
@testing.with_requires('scipy')
class TestPolyGamma(unittest.TestCase):

    def setUp(self):
        self.x = \
            numpy.random.uniform(1., 10., self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = \
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.n = numpy.random.randint(3, size=self.shape).astype(numpy.int32)
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-4}
        self.check_backward_options = {'eps': 1e-3, 'atol': 5e-2, 'rtol': 1e-3,
                                       'dtype': numpy.float64}
        self.check_double_backward_options = {'eps': 1e-3, 'atol': 5e-2,
                                              'rtol': 1e-3,
                                              'dtype': numpy.float64}

    def check_forward(self, n_data, x_data):
        from scipy import special
        x = chainer.Variable(x_data)
        n = chainer.Variable(n_data)
        y = F.polygamma(n, x)
        testing.assert_allclose(
            special.polygamma(self.n, self.x), y.data,
            **self.check_forward_options)

    def test_polygamma_forward_cpu(self):
        self.check_forward(self.n, self.x)

    @attr.gpu
    def test_polygamma_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.n), cuda.to_gpu(self.x))

    def check_backward(self, n_data, x_data, y_grad):
        gradient_check.check_backward(
            lambda x: F.polygamma(chainer.Variable(n_data), x), x_data, y_grad,
            **self.check_backward_options)

    def test_polygamma_backward_cpu(self):
        self.check_backward(self.n, self.x, self.gy)

    @attr.gpu
    def test_polygamma_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.n), cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, n_data, x_data, y_grad, x_grad_grad):
        gradient_check.check_double_backward(
            lambda x: F.polygamma(chainer.Variable(n_data), x), x_data, y_grad,
            x_grad_grad, **self.check_double_backward_options)

    def test_polygamma_double_backward_cpu(self):
        self.check_double_backward(self.n, self.x, self.gy, self.ggx)

    @attr.gpu
    def test_polygamma_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.n), cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))


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

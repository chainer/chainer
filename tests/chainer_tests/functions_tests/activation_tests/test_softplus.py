import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
class TestSoftplus(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.beta = numpy.random.uniform(1, 2, ())
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.softplus(x, beta=self.beta)
        x_value = cuda.to_cpu(x_data)
        y_exp = numpy.log(1 + numpy.exp(self.beta * x_value)) / self.beta
        self.assertEqual(y.data.dtype, self.dtype)
        testing.assert_allclose(
            y_exp, y.data, **self.check_forward_options)

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.softplus(x, beta=self.beta)
        gradient_check.check_backward(
            f, x_data, y_grad,
            dtype=numpy.float64, **self.check_backward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            return functions.softplus(x, beta=self.beta)

        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad,
                dtype=numpy.float64, **self.check_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)

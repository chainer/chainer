import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
class TestClippedReLU(unittest.TestCase):

    def setUp(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Avoid values around zero and z for stability of numerical gradient
        x[((-0.01 < x) & (x < 0.01)) | ((0.74 < x) & (x < 0.76))] = 0.5
        self.x = x

        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, x.shape).astype(self.dtype)
        self.z = 0.75

        if self.dtype == numpy.float16:
            self.check_backward_options = {}
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        else:
            self.check_backward_options = {}
            self.check_double_backward_options = {}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.clipped_relu(x, self.z)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = self.x.clip(0, self.z)

        testing.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.clipped_relu(x, self.z)

        gradient_check.check_backward(
            f, x_data, y_grad, dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            y = functions.clipped_relu(x, self.z)
            return y * y

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)

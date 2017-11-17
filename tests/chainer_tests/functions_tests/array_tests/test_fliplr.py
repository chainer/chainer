import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 4), (3, 4, 2)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestFlipLR(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.fliplr(x)

        testing.assert_allclose(y.data, numpy.fliplr(self.x))

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.fliplr, x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            y = functions.fliplr(x)
            return y * y

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
            atol=5e-4, rtol=5e-3)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)

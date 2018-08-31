import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(4, 3), (4, 3, 2), (4,), (), (1,), (1, 1)],
}))
class TestSquaredError(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x1 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.random(self.shape).astype(self.dtype)
        self.ggx0 = numpy.random.uniform(-1, 1, self.shape) \
            .astype(self.dtype)
        self.ggx1 = numpy.random.uniform(-1, 1, self.shape) \
            .astype(self.dtype)

        if self.dtype == numpy.float16:
            self.places = 2
            self.check_backward_options = {'atol': 5e-1, 'rtol': 5e-1}
            self.check_double_backward_options = {'atol': 4e-1, 'rtol': 4e-1}
        else:
            self.places = 5
            self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_double_backward_options = {}

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.squared_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        assert loss_value.dtype == self.dtype
        assert loss_value.shape == x0_data.shape

        for i in numpy.ndindex(self.x0.shape):
            # Compute expected value
            loss_expect = (self.x0[i] - self.x1[i]) ** 2
            assert round(loss_value[i] - loss_expect, self.places) == 0

    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data, y_grad):
        gradient_check.check_backward(
            functions.squared_error,
            (x0_data, x1_data), y_grad, eps=1e-2,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1), cuda.to_gpu(self.gy))

    def check_double_backward(self, x0_data, x1_data, y_grad, ggx0_data,
                              ggx1_data):
        gradient_check.check_double_backward(
            functions.squared_error,
            (x0_data, x1_data), y_grad, (ggx0_data, ggx1_data), eps=1e-2,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x0, self.x1, self.gy, self.ggx0, self.ggx1)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx0), cuda.to_gpu(self.ggx1))


testing.run_module(__name__, __file__)

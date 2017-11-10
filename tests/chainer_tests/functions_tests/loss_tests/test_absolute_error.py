import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'shape': (4, 3)},
    {'shape': (4, 3, 2)},
    {'shape': (4,)},
    {'shape': ()},
    {'shape': (1,)},
    {'shape': (1, 1)},
)
class TestAbsoluteError(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        # Add sufficient margin to prevent computational error
        diff = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        diff[abs(diff) < 0.01] = 0.5
        self.x1 = numpy.asarray(self.x0 + diff)
        self.gy = numpy.random.random(self.shape).astype(numpy.float32)
        self.ggx0 = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)
        self.ggx1 = numpy.random.uniform(
            -1, 1, self.shape).astype(numpy.float32)

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.absolute_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        assert loss_value.dtype == numpy.float32
        assert loss_value.shape == x0_data.shape

        for i in numpy.ndindex(self.x0.shape):
            # Compute expected value
            loss_expect = abs(self.x0[i] - self.x1[i])
            assert round(loss_value[i] - loss_expect, 5) == 0

    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data, y_grad):
        gradient_check.check_backward(
            functions.absolute_error,
            (x0_data, x1_data), y_grad, dtype='d')

    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1), cuda.to_gpu(self.gy))

    def check_double_backward(self, x0_data, x1_data, y_grad,
                              gx0_grad, gx1_grad):
        def f(x0, x1):
            y = functions.absolute_error(x0, x1)
            return y * y

        gradient_check.check_double_backward(
            f, (x0_data, x1_data), y_grad, (gx0_grad, gx1_grad), eps=1e-2)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x0, self.x1, self.gy, self.ggx0, self.ggx1)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx0), cuda.to_gpu(self.ggx1))


testing.run_module(__name__, __file__)

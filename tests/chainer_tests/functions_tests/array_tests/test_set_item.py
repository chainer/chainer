import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import parameterize


@parameterize(
    {'shape': (4, 3, 2), 'slices': (1, -1), 'batch_ndim': 0},
    {'shape': (4, 3, 2), 'slices': (1, -1), 'batch_ndim': 1},
)
class TestSetItem(unittest.TestCase):

    def setUp(self):
        self.x0_data = numpy.random.uniform(-1, 1, self.shape)
        sliced_shape = self.x0_data[self.slices].shape
        rhs_shape = sliced_shape[self.batch_ndim:]
        self.x1_data = numpy.random.uniform(-1, 1, rhs_shape)
        self.gy_data = numpy.random.uniform(-1, 1, self.shape)
        self.ggx0_data = numpy.random.uniform(-1, 1, self.shape)
        self.ggx1_data = numpy.random.uniform(-1, 1, rhs_shape)

    def _forward(self, x0, x1):
        return functions.copied_set_item(x0, self.slices, x1)

    def check_forward(self, x0_data, x1_data):
        y_expected = x0_data.copy()
        y_expected[self.slices] = x1_data

        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        y = self._forward(x0, x1)
        testing.assert_allclose(y.array, y_expected)

    def test_forward_cpu(self):
        self.check_forward(self.x0_data, self.x1_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x0_data))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self._forward, x_data, y_grad)

    def test_backward_cpu(self):
        self.check_backward((self.x0_data, self.x1_data), self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            (cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x1_data)),
            cuda.to_gpu(self.gy_data))

    def check_double_backward(self, x_data, y_grad, ggx_data):
        gradient_check.check_double_backward(
            self._forward, x_data, y_grad, ggx_data)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            (self.x0_data, self.x1_data), self.gy_data,
            (self.ggx0_data, self.ggx1_data))

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            (cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x1_data)),
            cuda.to_gpu(self.gy_data),
            (cuda.to_gpu(self.ggx0_data), cuda.to_gpu(self.ggx1_data)))


testing.run_module(__name__, __file__)

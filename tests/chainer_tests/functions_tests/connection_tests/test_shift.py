import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainer.functions.connection import shift


@testing.parameterize(*(testing.product({
    'shape': [(4, 3)],
    'c_contiguous': [True, False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batches': [1, 2],
    'ksize': [(3, 3), (3, 1), (1, 3), (1, 5), (5, 1)],
    'n_channels': [9, 16, 25],
    'dilate': [1],
}) + testing.product({
    'shape': [(10, 8)],
    'c_contiguous': [True, False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batches': [1, 2],
    'ksize': [(5, 5)],
    'n_channels': [25],
    'dilate': [1],
}) + testing.product({
    'shape': [(10, 8)],
    'c_contiguous': [True, False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batches': [1, 2],
    'ksize': [(7, 7), (7, 1), (1, 7)],
    'n_channels': [100],
    'dilate': [1],
}) + testing.product({
    'shape': [(4, 3)],
    'c_contiguous': [True, False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batches': [1, 2],
    'ksize': [(3, 3), (3, 1), (1, 3), (1, 5), (5, 1)],
    'n_channels': [16],
    'dilate': [2, 3],
})))
class TestShiftFunction(unittest.TestCase):

    def setUp(self):
        h, w = self.shape
        self.x = numpy.random.uniform(
            -1, 1, (self.batches, self.n_channels, h, w)).astype(self.x_dtype)
        self.gy = numpy.random.uniform(
            -1, 1, (self.batches, self.n_channels, h, w)).astype(self.x_dtype)

    @attr.gpu
    def test_forward_consistency(self):
        x_data = self.x
        xp = backend.get_array_module(x_data)

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            self.assertFalse(x_data.flags.c_contiguous)

        x_cpu = chainer.Variable(x_data)
        y_cpu = shift.shift(
            x_cpu, ksize=self.ksize, dilate=self.dilate)

        x_gpu = chainer.Variable(cuda.to_gpu(x_data))
        y_gpu = shift.shift(
            x_gpu, ksize=self.ksize, dilate=self.dilate)

        testing.assert_allclose(
            y_cpu.data, y_gpu.data.get(), atol=5e-4, rtol=5e-3)

    def check_backward(self, x_data, y_grad):
        xp = backend.get_array_module(x_data)

        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)

        gradient_check.check_backward(
            lambda x: shift.shift(x, ksize=self.ksize, dilate=self.dilate),
            x_data, y_grad, dtype='d', atol=5e-4, rtol=5e-3)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

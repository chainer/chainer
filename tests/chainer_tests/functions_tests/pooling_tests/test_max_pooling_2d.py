import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestMaxPooling2D(unittest.TestCase):
    cover_all = False

    def setUp(self):
        # Avoid unstability of numerical gradient
        self.x = numpy.arange(
            2 * 3 * 4 * 3, dtype=numpy.float32).reshape(2, 3, 4, 3)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1

        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 3, 2, 2)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                     cover_all=self.cover_all,
                                     use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                if self.cover_all:
                    expect = numpy.array([
                        [self.x[k, c, 0:2, 0:2].max(), self.x[
                            k, c, 0:2, 1:3].max()],
                        [self.x[k, c, 1:4, 0:2].max(), self.x[
                            k, c, 1:4, 1:3].max()],
                        [self.x[k, c, 3:4, 0:2].max(), self.x[
                            k, c, 3:4, 1:3].max()]])
                else:
                    expect = numpy.array([
                        [self.x[k, c, 0:2, 0:2].max(), self.x[
                            k, c, 0:2, 1:3].max()],
                        [self.x[k, c, 1:4, 0:2].max(), self.x[
                            k, c, 1:4, 1:3].max()]])
                gradient_check.assert_allclose(expect, y_data[k, c])

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    def test_forward_cpu_wide(self):  # see #120
        x_data = numpy.random.rand(2, 3, 15, 15).astype(numpy.float32)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.MaxPooling2D(
                3, stride=2, pad=1, cover_all=self.cover_all,
                use_cudnn=use_cudnn),
            x_data, y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)


class TestMaxPooling2DCoverAll(TestMaxPooling2D):
    cover_all = True

    def setUp(self):
        super(TestMaxPooling2DCoverAll, self).setUp()
        self.gy = numpy.random.uniform(-1, 1,
                                       (2, 3, 3, 2)).astype(numpy.float32)


testing.run_module(__name__, __file__)

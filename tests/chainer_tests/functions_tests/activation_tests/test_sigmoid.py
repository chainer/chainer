import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestSigmoid(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-.5, .5, (3, 2)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-.1, .1, (3, 2)).astype(numpy.float32)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self, use_cudnn=True):
        x = chainer.Variable(cuda.to_gpu(self.x))
        y = functions.sigmoid(x, use_cudnn=use_cudnn)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = functions.sigmoid(chainer.Variable(self.x))

        gradient_check.assert_allclose(y_expect.data, y.data)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.test_forward_gpu(False)

    def check_backward(self, x_data, y_grad, use_cudnn=True):
        gradient_check.check_backward(
            functions.Sigmoid(use_cudnn), x_data, y_grad)

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


testing.run_module(__name__, __file__)

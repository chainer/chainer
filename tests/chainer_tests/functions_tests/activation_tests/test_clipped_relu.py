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
        self.z = 0.75

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
        gradient_check.check_backward(
            functions.ClippedReLU(self.z), x_data, y_grad, dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

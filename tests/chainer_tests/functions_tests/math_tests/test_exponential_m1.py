import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(), (3, 2)],
}))
class Expm1FunctionTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.ggx = \
            numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)

        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = F.expm1(x)
        testing.assert_allclose(
            numpy.expm1(self.x), y.data, atol=1e-7, rtol=1e-7)

    def test_expm1_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_expm1_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            F.expm1, x_data, y_grad,
            **self.check_backward_options)

    def test_expm1_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_expm1_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        gradient_check.check_double_backward(
            F.expm1, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def test_expm1_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_expm1_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    def test_expm1(self):
        self.assertEqual(
            chainer.functions.math.exponential_m1.Expm1().label, 'expm1')


testing.run_module(__name__, __file__)

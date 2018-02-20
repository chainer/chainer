import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class UnaryFunctionsTestBase(unittest.TestCase):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.eps = 1e-3
        while True:
            self.x, self.gy, self.ggy = self.make_data()
            if (numpy.abs(self.x - numpy.round(self.x)) > self.eps * 10).all():
                break

    def check_forward(self, op, op_xp, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        self.assertEqual(x.data.dtype, y.data.dtype)
        v = op_xp(x_data)
        testing.assert_allclose(
            v, y.data, atol=1e-7, rtol=1e-7)

    def check_forward_cpu(self, op, op_xp):
        self.check_forward(op, op_xp, self.x)

    def check_forward_gpu(self, op, op_xp):
        self.check_forward(op, op_xp, cuda.to_gpu(self.x))

    def check_backward(self, op, x_data, y_grad):
        gradient_check.check_backward(op, x_data, y_grad, atol=5e-4,
                                      rtol=5e-3, dtype=numpy.float64,
                                      eps=self.eps)

    def check_backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def check_backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, op, x_data, y_grad, y_grad_grad):
        def f(x):
            x = op(x)
            return x * x
        gradient_check.check_double_backward(
            f, x_data, y_grad, y_grad_grad, dtype=numpy.float64,
            atol=1e-7, rtol=1e-7)

    def check_double_backward_cpu(self, op):
        self.check_double_backward(op, self.x, self.gy, self.ggy)

    def check_double_backward_gpu(self, op):
        self.check_double_backward(op, cuda.to_gpu(
            self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggy))

    def check_label(self, op, expected):
        self.assertEqual(op().label, expected)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestFloor(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-10.0, 10.0, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        ggy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy, ggy

    def test_forward_cpu(self):
        self.check_forward_cpu(F.floor, numpy.floor)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.floor, cuda.cupy.floor)


testing.run_module(__name__, __file__)

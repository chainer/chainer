import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class UnaryFunctionsTestBase(unittest.TestCase):

    @property
    def no_grads(self):
        return (False,)

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.eps = 1e-3
        while True:
            self.x, self.gy, self.ggx = self.make_data()
            if (numpy.abs(self.x - numpy.round(self.x)) > self.eps * 10).all():
                break

    def check_forward(self, op, op_xp, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        v = op_xp(x_data)

        assert x.data.dtype == y.data.dtype
        testing.assert_allclose(v, y.data, atol=1e-7, rtol=1e-7)

    def check_forward_ndarray(self, op, op_xp, x_data):
        y = op(x_data)
        v = op_xp(x_data)

        assert x_data.dtype == y.data.dtype
        assert x_data.dtype == v.dtype
        assert isinstance(y, chainer.Variable)
        testing.assert_allclose(v, y.data, atol=1e-7, rtol=1e-7)

    def check_forward_cpu(self, op, op_xp):
        self.check_forward(op, op_xp, self.x)
        self.check_forward_ndarray(op, op_xp, self.x)

    def check_forward_gpu(self, op, op_xp):
        self.check_forward(op, op_xp, cuda.to_gpu(self.x))
        self.check_forward_ndarray(op, op_xp, cuda.to_gpu(self.x))

    def check_backward(self, op, x_data, y_grad, no_grads):
        gradient_check.check_backward(op, x_data, y_grad, no_grads=no_grads)

        if no_grads and no_grads[0]:
            x = chainer.Variable(x_data)
            op(x).backward()
            assert x.grad is None

    def check_backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy, self.no_grads)

    def check_backward_gpu(self, op):
        self.check_backward(
            op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy), self.no_grads)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSign(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-10.0, 10.0, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy, ggx

    @property
    def no_grads(self):
        return (True,)

    def test_forward_cpu(self):
        self.check_forward_cpu(F.sign, numpy.sign)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.sign, cuda.cupy.sign)

    def test_backward_cpu(self):
        self.check_backward_cpu(F.sign)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward_gpu(F.sign)


testing.run_module(__name__, __file__)

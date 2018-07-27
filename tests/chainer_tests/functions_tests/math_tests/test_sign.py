import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSign(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        # Avoid non-differentiable point
        self.x[(abs(self.x) < 1e-2)] = 1

    def check_forward(self, x_data, xp):
        x = chainer.Variable(x_data)
        y = F.sign(x)
        v = xp.sign(x_data)

        assert x.data.dtype == y.data.dtype
        testing.assert_allclose(v, y.data, atol=1e-7, rtol=1e-7)

    def test_forward_cpu(self):
        self.check_forward(self.x, numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.cupy)

    def check_forward_ndarray(self, x_data, xp):
        y = F.sign(x_data)
        v = xp.sign(x_data)

        assert x_data.dtype == y.data.dtype
        assert x_data.dtype == v.dtype
        assert isinstance(y, chainer.Variable)
        testing.assert_allclose(v, y.data, atol=1e-7, rtol=1e-7)

    def test_forward_ndarray_cpu(self):
        self.check_forward_ndarray(self.x, numpy)

    @attr.gpu
    def test_forward_ndarray_gpu(self):
        self.check_forward_ndarray(cuda.to_gpu(self.x), cuda.cupy)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            F.sign, x_data, y_grad)

        # Explicitly check that gradients are `None`
        x = chainer.Variable(x_data)
        F.sign(x).backward()
        assert x.grad is None

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

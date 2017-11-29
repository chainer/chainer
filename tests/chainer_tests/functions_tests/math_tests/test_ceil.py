import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import attr


class UnaryFunctionsTestBase(unittest.TestCase):

    def make_data(self):
        raise NotImplementedError

    def setUp(self):
        self.eps = 1e-3
        while True:
            self.x, self.gy = self.make_data()
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


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestCeil(UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-10.0, 10.0, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    def test_forward_cpu(self):
        self.check_forward_cpu(F.ceil, numpy.ceil)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward_gpu(F.ceil, cuda.cupy.ceil)


testing.run_module(__name__, __file__)

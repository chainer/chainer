import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _rrelu(x, creator, train):
    r = creator.r if train else (creator.lower + creator.upper) / 2
    return x * numpy.where(x < 0, r, 1)


@testing.parameterize(*testing.product({
    'train': [True, False],
    'shape': [(3, 2), (5, 6)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numeraical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if -0.05 < self.x[i] < 0.05:
                self.x[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Asummption l < u
        self.l = numpy.random.uniform(0, 1)
        self.u = numpy.random.uniform(0, 1)
        if self.l >= self.u:
            self.l, self.u = self.u, self.l
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        chainer.config.train = self.train
        y = functions.rrelu(x, l=self.l, u=self.u)
        self.assertEqual(y.data.dtype, self.dtype)

        expected = self.x.copy()
        for i in numpy.ndindex(self.x.shape):
            if self.x[i] < 0:
                expected[i] *= y.creator.r[i] if self.train else (
                    self.l + self.u) / 2

        testing.assert_allclose(
            expected, y.data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        chainer.config.train = self.train
        y = functions.rrelu(x, self.l, self.u)
        creator = y.creator
        y.grad = y_grad
        y.backward()

        def f():
            y = _rrelu(x_data, creator, self.train)
            return y,
        gx, = gradient_check.numerical_grad(f, (x_data, ), (y.grad, ), eps=0.1)
        testing.assert_allclose(gx, x.grad, **self.check_backward_options)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

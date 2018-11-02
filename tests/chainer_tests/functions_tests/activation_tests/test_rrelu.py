import unittest

import numpy

import chainer
from chainer import backend
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'train': [True, False],
    'shape': [(3, 2), (5, 6)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[(-0.05 < self.x) & (self.x < 0.05)] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Assumption l < u
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
        xp = backend.get_array_module(x)
        with chainer.using_config('train', self.train):
            y, r = functions.rrelu(x, l=self.l, u=self.u, return_r=True)
        self.assertEqual(y.data.dtype, self.dtype)
        expected = xp.where(x_data >= 0, x_data, x_data * r)
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
        xp = backend.get_array_module(x_data)
        r = xp.random.uniform(self.l, self.u, x_data.shape).astype(
            x_data.dtype)

        def f(x):
            return functions.rrelu(x, self.l, self.u, r=r)

        with chainer.using_config('train', self.train):
            gradient_check.check_backward(
                f, x_data, y_grad, dtype=numpy.float64,
                **self.check_backward_options)

    @condition.retry(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'specify_r': [True, False],
    'train': [True, False],
    'shape': [(3, 2), (5, 6)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRReLUR(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[(-0.05 < self.x) & (self.x < 0.05)] = 0.5
        # Assumption l < u
        self.l = numpy.random.uniform(0, 1)
        self.u = numpy.random.uniform(0, 1)
        if self.l >= self.u:
            self.l, self.u = self.u, self.l
        self.r = numpy.random.uniform(
            self.l, self.u, self.x.shape).astype(self.x.dtype)

    def _check(self):
        r = self.r if self.specify_r else None
        with chainer.using_config('train', self.train):
            out, out_r = functions.rrelu(
                self.x, self.l, self.u, r=r, return_r=True)

        assert isinstance(out_r, type(out.array))
        if r is None:
            assert out_r.shape == out.array.shape
        else:
            if self.train:
                assert out_r is r

    def test_cpu(self):
        with chainer.using_config('use_ideep', 'never'):
            self._check()

    @attr.gpu
    def test_gpu(self):
        self.x = cuda.to_gpu(self.x)
        self.r = cuda.to_gpu(self.r)
        self._check()


testing.run_module(__name__, __file__)

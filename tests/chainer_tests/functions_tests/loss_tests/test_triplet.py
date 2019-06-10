import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batchsize': [5, 10],
    'input_dim': [2, 3],
    'margin': [0.1, 0.5],
    'reduce': ['mean', 'no']
}))
class TestTriplet(unittest.TestCase):

    def setUp(self):
        eps = 1e-3
        x_shape = (self.batchsize, self.input_dim)

        # Sample differentiable inputs
        while True:
            self.a = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            self.p = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            self.n = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            if (abs(self.a - self.p) < 2 * eps).any():
                continue
            if (abs(self.a - self.n) < 2 * eps).any():
                continue
            dist = numpy.sum(
                (self.a - self.p) ** 2 - (self.a - self.n) ** 2,
                axis=1) + self.margin
            if (abs(dist) < 2 * eps).any():
                continue
            break

        if self.reduce == 'mean':
            gy_shape = ()
        else:
            gy_shape = self.batchsize,
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        self.gga = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.ggp = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.ggn = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'rtol': 5e-3, 'atol': 5e-3}
            self.check_backward_options = {'rtol': 5e-2, 'atol': 5e-2}
            self.check_double_backward_options = {
                'dtype': numpy.float64, 'rtol': 1e-3, 'atol': 1e-3}
        elif self.dtype == numpy.float32:
            self.check_forward_options = {'rtol': 1e-4, 'atol': 1e-4}
            self.check_backward_options = {'rtol': 5e-4, 'atol': 5e-4}
            self.check_double_backward_options = {'rtol': 1e-3, 'atol': 1e-3}
        elif self.dtype == numpy.float64:
            self.check_forward_options = {'rtol': 1e-4, 'atol': 1e-4}
            self.check_backward_options = {'rtol': 5e-4, 'atol': 5e-4}
            self.check_double_backward_options = {'rtol': 1e-3, 'atol': 1e-3}
        else:
            raise ValueError('invalid dtype')

    def check_forward(self, a_data, p_data, n_data):
        a_val = chainer.Variable(a_data)
        p_val = chainer.Variable(p_data)
        n_val = chainer.Variable(n_data)
        loss = functions.triplet(a_val, p_val, n_val, self.margin, self.reduce)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            self.assertEqual(loss.data.shape, (self.batchsize,))
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = cuda.to_cpu(loss.data)

        #
        # Compute expected value
        #
        loss_expect = numpy.empty((self.a.shape[0],), dtype=self.dtype)
        for i in six.moves.range(self.a.shape[0]):
            ad, pd, nd = self.a[i], self.p[i], self.n[i]
            dp = numpy.sum((ad - pd) ** 2)
            dn = numpy.sum((ad - nd) ** 2)
            loss_expect[i] = max((dp - dn + self.margin), 0)
        if self.reduce == 'mean':
            loss_expect = loss_expect.mean()
        numpy.testing.assert_allclose(
            loss_expect, loss_value, **self.check_forward_options)

    def test_negative_margin(self):
        self.margin = -1
        self.assertRaises(ValueError, self.check_forward,
                          self.a, self.p, self.n)
        self.assertRaises(ValueError, self.check_backward,
                          self.a, self.p, self.n, self.gy)

    def test_forward_cpu(self):
        self.check_forward(self.a, self.p, self.n)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.a), cuda.to_gpu(self.p),
                           cuda.to_gpu(self.n))

    def check_backward(self, a_data, p_data, n_data, gy_data):
        def f(a, p, n):
            return functions.triplet(
                a, p, n, margin=self.margin, reduce=self.reduce)

        gradient_check.check_backward(
            f, (a_data, p_data, n_data), gy_data, dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.a, self.p, self.n, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.a), cuda.to_gpu(self.p),
                            cuda.to_gpu(self.n), cuda.to_gpu(self.gy))

    def check_double_backward(self, a_data, p_data, n_data, gy_data, gga_data,
                              ggp_data, ggn_data):
        def f(a, p, n):
            return functions.triplet(
                a, p, n, margin=self.margin, reduce=self.reduce)

        gradient_check.check_double_backward(
            f, (a_data, p_data, n_data), gy_data,
            (gga_data, ggp_data, ggn_data),
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.a, self.p, self.n, self.gy, self.gga, self.ggp, self.ggn)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.a), cuda.to_gpu(self.p), cuda.to_gpu(self.n),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.gga), cuda.to_gpu(self.ggp),
            cuda.to_gpu(self.ggn))


class TestContrastiveInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.a = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.p = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.n = numpy.random.randint(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        a = xp.asarray(self.a)
        p = xp.asarray(self.p)
        n = xp.asarray(self.n)

        with self.assertRaises(ValueError):
            functions.triplet(a, p, n, reduce='invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

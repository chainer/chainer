import unittest

import numpy as np
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


def _as_two_dim(x):
    if x.ndim == 2:
        return x
    return x.reshape((len(x), -1))


@testing.parameterize(*testing.product({
    'dtype': [np.float16, np.float32, np.float64],
    'shape': [(4, 3, 5), (4, 15)],
}))
class TestBatchL2NormSquared(unittest.TestCase):

    def setUp(self):
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.gy = np.random.uniform(-1, 1, self.shape[0]).astype(self.dtype)
        self.ggx = np.random.uniform(-1, 1, self.shape).astype(self.dtype)

        if self.dtype == np.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {}
            self.check_double_backward_options = {}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.batch_l2_norm_squared(x)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        x_two_dim = _as_two_dim(self.x)
        y_expect = np.empty(len(self.x), dtype=self.dtype)
        for n in six.moves.range(len(self.x)):
            y_expect[n] = sum(map(lambda x: x * x, x_two_dim[n]))

        testing.assert_allclose(y_expect, y_data, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.batch_l2_norm_squared, x_data, y_grad, eps=1,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        gradient_check.check_double_backward(
            functions.batch_l2_norm_squared,
            x_data, y_grad, x_grad_grad, dtype=np.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))


class TestBatchL2NormSquaredTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x = chainer.Variable(np.zeros((4,), dtype=np.float32))

        with self.assertRaises(type_check.InvalidType):
            chainer.functions.batch_l2_norm_squared(x)


testing.run_module(__name__, __file__)

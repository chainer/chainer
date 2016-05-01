import unittest

import numpy as np
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(
    {'shape': (4, 15)},
    {'shape': (4, 15)},
)
class TestL2Normalization(unittest.TestCase):

    def setUp(self):
        self.x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.gy = np.random.uniform(-1, 1, self.shape).astype(np.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.l2_normalization(x)
        self.assertEqual(y.data.dtype, np.float32)
        y_data = cuda.to_cpu(y.data)

        y_expect = np.empty_like(self.x)
        for n in six.moves.range(len(self.x)):
            y_expect[n] = self.x[n]/np.linalg.norm(self.x[n])

        gradient_check.assert_allclose(y_expect, y_data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.L2Normalization(), x_data, y_grad, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestL2NormalizationTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x = chainer.Variable(np.zeros((4, 3, 24), dtype=np.float32))

        with self.assertRaises(type_check.InvalidType):
            chainer.functions.l2_normalization(x)


testing.run_module(__name__, __file__)

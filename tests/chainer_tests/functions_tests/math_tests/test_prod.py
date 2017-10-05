import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'axis': [None, 0, 1, 2, -1, (0, 1), (1, 0), (0, -1), (-2, 0)],
    'keepdims': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'contain_zero': [True, False],
}))
class TestProd(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        if self.contain_zero:
            index = numpy.random.choice(self.x.size)
            self.x.ravel()[index] = 0
        g_shape = self.x.prod(axis=self.axis, keepdims=self.keepdims).shape
        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.prod(x, axis=self.axis, keepdims=self.keepdims)
        self.assertEqual(y.data.dtype, self.dtype)
        y_expect = self.x.prod(axis=self.axis, keepdims=self.keepdims)

        if self.dtype == numpy.float16:
            options = {'atol': 1e-3, 'rtol': 1e-3}
        else:
            options = {}

        testing.assert_allclose(y_expect, y.data, **options)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.Prod(self.axis, self.keepdims), x_data, y_grad,
            atol=1e-3, dtype=numpy.float64)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_axis_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestProdError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.Prod([0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.Prod((1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.Prod((0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.x.prod(axis=(1, -2))


testing.run_module(__name__, __file__)

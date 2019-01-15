import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'shape': [
        # x1, x2, y
        ((3, 2), (3, 2), (3, 2)),
        ((), (), ()),
        ((3, 2), (3, 1), (3, 2)),
        ((2,), (3, 2), (3, 2)),
    ],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestMinimum(unittest.TestCase):

    def setUp(self):
        x1_shape, x2_shape, y_shape = self.shape
        self.gy = numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)
        self.ggx1 = numpy.random.uniform(-1, 1, x1_shape).astype(self.dtype)
        self.ggx2 = numpy.random.uniform(-1, 1, x2_shape).astype(self.dtype)
        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        self.check_double_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            eps = 1e-2
            self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
            self.check_backward_options.update({
                'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update({
                'atol': 1e-2, 'rtol': 1e-2})
        else:
            eps = 1e-3
        self.check_backward_options['eps'] = eps
        self.check_double_backward_options['eps'] = eps

        self.x1 = numpy.random.uniform(-1, 1, x1_shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(-1, 1, x2_shape).astype(self.dtype)

        # Avoid close values for stability in numerical gradient.
        retry = 0
        while True:
            self.x1 = numpy.random.uniform(-1, 1, x1_shape).astype(self.dtype)
            self.x2 = numpy.random.uniform(-1, 1, x2_shape).astype(self.dtype)
            if (abs(self.x1 - self.x2) >= 2 * eps).all():
                break
            retry += 1
            assert retry <= 10, 'Too many retries to generate inputs'

        self.y_expected = numpy.minimum(self.x1, self.x2)

    def check_forward(self, x1_data, x2_data, y_expected):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = functions.minimum(x1, x2)
        self.assertEqual(y.data.dtype, self.dtype)
        testing.assert_allclose(
            y_expected, y.data, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x1, self.x2, self.y_expected)

    @attr.gpu
    def test_forward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        self.check_forward(x1, x2, self.y_expected)

    def check_backward(self, x1_data, x2_data, y_grad):
        func = functions.minimum
        x = (x1_data, x2_data)
        gradient_check.check_backward(
            func, x, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x1, self.x2, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x1, x2, gy)

    def check_double_backward(self, x1, x2, gy, ggx1, ggx2):
        gradient_check.check_double_backward(
            functions.minimum, (x1, x2), gy, (ggx1, ggx2),
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x1, self.x2, self.gy, self.ggx1, self.ggx2)

    @attr.gpu
    def test_double_backward_gpu(self):
        x1 = cuda.to_gpu(self.x1)
        x2 = cuda.to_gpu(self.x2)
        gy = cuda.to_gpu(self.gy)
        ggx1 = cuda.to_gpu(self.ggx1)
        ggx2 = cuda.to_gpu(self.ggx2)
        self.check_double_backward(x1, x2, gy, ggx1, ggx2)


@testing.parameterize(*testing.product({
    'dtype1': [numpy.float16, numpy.float32, numpy.float64],
    'dtype2': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestMinimumInconsistentTypes(unittest.TestCase):

    def test_minimum_inconsistent_types(self):
        if self.dtype1 == self.dtype2:
            return
        x1_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype1)
        x2_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype2)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        with self.assertRaises(type_check.InvalidType):
            functions.minimum(x1, x2)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestMinimumInconsistentShapes(unittest.TestCase):

    def test_minimum_inconsistent_shapes(self):
        x1_data = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)
        x2_data = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        with self.assertRaises(type_check.InvalidType):
            functions.minimum(x1, x2)


testing.run_module(__name__, __file__)

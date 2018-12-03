import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(
    {'dtype': numpy.float32,
     'places': 5,
     'backward_tols': {'atol': 1e-5, 'rtol': 1e-4},
     'double_backward_tols': {'atol': 1e-5, 'rtol': 1e-4}},
    {'dtype': numpy.float16,
     'places': 3,
     'backward_tols': {'atol': 5e-2, 'rtol': 5e-1},
     'double_backward_tols': {'atol': 5e-2, 'rtol': 5e-1}},
)
class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        self.x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        self.x1 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(dtype)
        self.ggx0 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        self.ggx1 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.mean_squared_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, self.dtype)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.
        for i in numpy.ndindex(self.x0.shape):
            loss_expect += (self.x0[i] - self.x1[i]) ** 2
        loss_expect /= self.x0.size

        self.assertAlmostEqual(loss_expect, loss_value, places=self.places)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data):
        gradient_check.check_backward(
            functions.mean_squared_error,
            (x0_data, x1_data), None, eps=1e-2, **self.backward_tols)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_double_backward(self, x0_data, x1_data, gy_data,
                              ggx0_data, ggx1_data):
        gradient_check.check_double_backward(
            functions.mean_squared_error, (x0_data, x1_data), gy_data,
            (ggx0_data, ggx1_data), eps=1e-2, **self.double_backward_tols)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x0, self.x1, self.gy, self.ggx0, self.ggx1)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx0), cuda.to_gpu(self.ggx1))


class TestMeanSquaredErrorTypeCheck(unittest.TestCase):

    def test_invalid_dtype1(self):
        x0 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        x1 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_squared_error(x0, x1)

    def test_invalid_dtype2(self):
        x0 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32))
        x1 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float16))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_squared_error(x0, x1)


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(
    {'dtype': numpy.float32,
     'ndigits': 5,
     'backward_tols': {'atol': 1e-5, 'rtol': 1e-4},
     'double_backward_tols': {'atol': 1e-5, 'rtol': 1e-4}},
    {'dtype': numpy.float16,
     'ndigits': 3,
     'backward_tols': {'atol': 5e-2, 'rtol': 5e-1},
     'double_backward_tols': {'atol': 5e-2, 'rtol': 5e-1}},
)
class TestMeanAbsoluteError(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        self.x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        # Add sufficient margin to prevent computational error
        diff = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        diff[abs(diff) < 0.01] = 0.5
        self.x1 = self.x0 + diff
        self.gy = numpy.random.uniform(-1, 1, ()).astype(dtype)
        self.ggx0 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)
        self.ggx1 = numpy.random.uniform(-1, 1, (4, 3)).astype(dtype)

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.mean_absolute_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)

        assert loss_value.dtype == self.dtype
        assert loss_value.shape == ()

        # Compute expected value
        loss_expect = 0.
        for i in numpy.ndindex(self.x0.shape):
            loss_expect += abs(self.x0[i] - self.x1[i])
        loss_expect /= self.x0.size

        assert round(loss_expect - loss_value, self.ndigits) == 0

    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data):
        gradient_check.check_backward(
            functions.mean_absolute_error,
            (x0_data, x1_data), None, eps=1e-2, **self.backward_tols)

    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_double_backward(self, x0_data, x1_data, gy_data, ggx0_data,
                              ggx1_data):
        gradient_check.check_double_backward(
            chainer.functions.mean_absolute_error, (x0_data, x1_data), gy_data,
            (ggx0_data, ggx1_data), eps=1e-2, **self.double_backward_tols)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x0, self.x1, self.gy, self.ggx0, self.ggx1)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x0),
                                   cuda.to_gpu(self.x1),
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx0),
                                   cuda.to_gpu(self.ggx1))

    # test for #4669
    @attr.multi_gpu(2)
    def test_backward_non_default_gpu(self):
        x0 = chainer.Variable(cuda.to_gpu(self.x0, 1))
        x1 = chainer.Variable(cuda.to_gpu(self.x1, 1))
        gy = cuda.to_gpu(self.gy, 1)
        with cuda.get_device_from_id(0):
            y = functions.mean_absolute_error(x0, x1)
            y.grad = gy
            y.backward()


class TestMeanAbsoluteErrorTypeCheck(unittest.TestCase):

    def test_invalid_dtype1(self):
        x0 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        x1 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.int32))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_absolute_error(x0, x1)

    def test_invalid_dtype2(self):
        x0 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32))
        x1 = chainer.Variable(
            numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float16))
        with self.assertRaises(type_check.InvalidType):
            functions.mean_absolute_error(x0, x1)


testing.run_module(__name__, __file__)

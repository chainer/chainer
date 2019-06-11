import unittest

import numpy
import pytest

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'shape': [
        # c, x, y, output
        ((3, 2, 4),) * 4,
        ((4,), (3, 1, 1), (2, 1), (3, 2, 4)),
    ],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestWhere(unittest.TestCase):

    def setUp(self):
        c_shape, x_shape, y_shape, out_shape = self.shape
        self.c_data = numpy.random.uniform(-1, 1, c_shape) > 0
        self.x_data = \
            numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.y_data = \
            numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)
        self.g_data = \
            numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-3,
            })

    def check_forward(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        z = functions.where(c, x, y)

        xp = c.xp
        z_data_expected = xp.where(c_data, x_data, y_data)
        testing.assert_allclose(z.array, z_data_expected)

    def test_forward_cpu(self):
        self.check_forward(self.c_data, self.x_data, self.y_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_data),
                           cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.y_data))

    def check_backward(self, c_data, x_data, y_data, g_data):
        gradient_check.check_backward(
            functions.where, (c_data, x_data, y_data), g_data,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.c_data, self.x_data, self.y_data, self.g_data)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.c_data),
                            cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.y_data),
                            cuda.to_gpu(self.g_data))


class TestWhereTypeCheck(unittest.TestCase):

    def check_forward_raises(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        with pytest.raises(type_check.InvalidType):
            functions.where(c, x, y)

    def test_cond_int(self):
        c_data = numpy.zeros(3, numpy.int32)
        x_data = numpy.zeros(3, numpy.float32)
        y_data = numpy.zeros(3, numpy.float32)
        self.check_forward_raises(c_data, x_data, y_data)

    def test_xy_precision(self):
        c_data = numpy.zeros(3, numpy.bool_)
        x_data = numpy.zeros(3, numpy.float32)
        y_data = numpy.zeros(3, numpy.float64)
        self.check_forward_raises(c_data, x_data, y_data)

    def test_shape(self):
        c_data = numpy.zeros(3, numpy.bool_)
        x_data = numpy.zeros(1, numpy.float32)
        y_data = numpy.zeros(2, numpy.float32)
        self.check_forward_raises(c_data, x_data, y_data)


testing.run_module(__name__, __file__)

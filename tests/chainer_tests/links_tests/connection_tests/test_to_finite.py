import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'x_shape': (15, 10, 10), 'axis': 1, 'ndim': 2},
    {'x_shape': (15, 10, 10), 'axis': 1, 'ndim': 1},
    {'x_shape': (15, 10, 10, 25), 'axis': 3, 'ndim': 1},
)
class TestToFinite(unittest.TestCase):
    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, self.x_shape).astype(numpy.float32)

        is_nan_x = numpy.random.binomial(1, 0.2, self.x_shape).astype(bool)
        is_posinf_x = numpy.random.binomial(1, 0.2, self.x_shape).astype(bool)
        is_neginf_x = numpy.random.binomial(1, 0.2, self.x_shape).astype(bool)
        self.x[is_nan_x] = numpy.nan
        self.x[is_posinf_x] = numpy.inf
        self.x[is_neginf_x] = -numpy.inf

        self.gy = numpy.random.uniform(
            -1, 1, self.x_shape).astype(numpy.float32)

        self.link = links.ToFinite(axis=self.axis, ndim=self.ndim)

    def test_attribute_presence(self):
        self.link(self.x)
        self.assertTrue(hasattr(self.link, 'nan_x'))
        self.assertTrue(hasattr(self.link, 'posinf_x'))
        self.assertTrue(hasattr(self.link, 'neginf_x'))

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)

        xp = x.xp
        self.assertTrue(xp.isfinite(y.data).all())

        isfinite_x = xp.isfinite(x_data)
        isnan_x = xp.isnan(x_data)
        isnan_where = xp.where(isnan_x)[self.axis: self.axis+self.ndim]
        if xp == numpy:
            isposinf_x = xp.isposinf(x_data)
            isposinf_where = \
                xp.where(isposinf_x)[self.axis: self.axis+self.ndim]
            isneginf_x = xp.isneginf(x_data)
            isneginf_where = \
                xp.where(isneginf_x)[self.axis: self.axis+self.ndim]
        else:
            isposinf_x = xp.isinf(x_data) * (x_data > 0)
            isposinf_where = \
                xp.where(isposinf_x)[self.axis: self.axis+self.ndim]
            isneginf_x = xp.isinf(x_data) * (x_data <= 0)
            isneginf_where = \
                xp.where(isneginf_x)[self.axis: self.axis+self.ndim]

        testing.assert_allclose(y[isfinite_x].array, x_data[isfinite_x])
        testing.assert_allclose(
            y[isnan_x].array, self.link.nan_x.data[isnan_where])
        testing.assert_allclose(
            y[isposinf_x].array, self.link.posinf_x.data[isposinf_where])
        testing.assert_allclose(
            y[isneginf_x].array, self.link.neginf_x.data[isneginf_where])

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        self.check_forward(x)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self.link, (x_data,), y_grad, atol=1e-2)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, gy)


testing.run_module(__name__, __file__)

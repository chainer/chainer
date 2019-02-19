import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape_and_axis': [
        # x, nan_x, posinf_x, neginf_x, axis
        ((15, 10, 20), (10, 20), (10,), (), 1),
        ((15, 10, 20), (15), (), (15, 10), 0),
    ],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
class TestToFinite(unittest.TestCase):

    def setUp(self):
        x_shape, nan_x_shape, posinf_x_shape, neginf_x_shape, axis = \
            self.shape_and_axis
        self.axis = axis

        self.x_data = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        is_nan_x = numpy.random.binomial(1, 0.2, x_shape).astype(bool)
        is_posinf_x = numpy.random.binomial(1, 0.2, x_shape).astype(bool)
        is_neginf_x = numpy.random.binomial(1, 0.2, x_shape).astype(bool)
        self.x_data[is_nan_x] = numpy.nan
        self.x_data[is_posinf_x] = numpy.inf
        self.x_data[is_neginf_x] = -numpy.inf

        self.nan_x_data = numpy.random.uniform(
            -1, 1, nan_x_shape).astype(self.dtype)
        self.posinf_x_data = numpy.random.uniform(
            -1, 1, posinf_x_shape).astype(self.dtype)
        self.neginf_x_data = numpy.random.uniform(
            -1, 1, neginf_x_shape).astype(self.dtype)

        self.g_data = \
            numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-3,
            })

    def check_forward(self, x_data, nan_x_data, posinf_x_data, neginf_x_data):
        x = chainer.Variable(x_data)
        nan_x = chainer.Variable(nan_x_data)
        posinf_x = chainer.Variable(posinf_x_data)
        neginf_x = chainer.Variable(neginf_x_data)

        y = functions.to_finite(x, nan_x, posinf_x, neginf_x, self.axis)

        xp = x.xp

        self.assertTrue(xp.isfinite(y.data).all())

        isfinite_x = xp.isfinite(x_data)
        isnan_x = xp.isnan(x_data)
        isnan_where = xp.where(isnan_x)[self.axis: self.axis+nan_x_data.ndim]
        if xp == numpy:
            isposinf_x = xp.isposinf(x_data)
            isposinf_where = \
                xp.where(isposinf_x)[self.axis: self.axis+posinf_x.ndim]
            isneginf_x = xp.isneginf(x_data)
            isneginf_where = \
                xp.where(isneginf_x)[self.axis: self.axis+neginf_x.ndim]
        else:
            isposinf_x = xp.isinf(x_data) * (x_data > 0)
            isposinf_where = \
                xp.where(isposinf_x)[self.axis: self.axis+posinf_x.ndim]
            isneginf_x = xp.isinf(x_data) * (x_data <= 0)
            isneginf_where = \
                xp.where(isneginf_x)[self.axis: self.axis+neginf_x.ndim]

        testing.assert_allclose(y[isfinite_x].array, x_data[isfinite_x])
        testing.assert_allclose(y[isnan_x].array, nan_x_data[isnan_where])
        testing.assert_allclose(
            y[isposinf_x].array, posinf_x_data[isposinf_where])
        testing.assert_allclose(
            y[isneginf_x].array, neginf_x_data[isneginf_where])

    def test_forward_cpu(self):
        self.check_forward(self.x_data, self.nan_x_data, self.posinf_x_data,
                           self.neginf_x_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.nan_x_data),
                           cuda.to_gpu(self.posinf_x_data),
                           cuda.to_gpu(self.neginf_x_data),)

    def check_backward(self, x_data, nan_x_data, posinf_x_data, neginf_x_data,
                       g_data):
        def f(x, nan_x, posinf_x, neginf):
            return functions.to_finite(x, nan_x, posinf_x, neginf, self.axis)

        gradient_check.check_backward(
            f, (x_data, nan_x_data, posinf_x_data, neginf_x_data), g_data,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.nan_x_data, self.posinf_x_data,
                            self.neginf_x_data, self.g_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.nan_x_data),
                            cuda.to_gpu(self.posinf_x_data),
                            cuda.to_gpu(self.neginf_x_data),
                            cuda.to_gpu(self.g_data),)

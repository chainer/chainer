import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _cross_covariance(y, z, dtype):
    row = y.shape[1]
    col = z.shape[1]
    y, z = cuda.to_cpu(y), cuda.to_cpu(z)
    y_mean = y.mean(axis=0)
    z_mean = z.mean(axis=0)
    N = y.shape[0]
    loss_expect = numpy.zeros((row, col), dtype=dtype)
    for i in six.moves.xrange(row):
        for j in six.moves.xrange(col):
            for n in six.moves.xrange(N):
                loss_expect[i, j] += (y[n, i] - y_mean[i]) * (
                    z[n, j] - z_mean[j])
    loss_expect /= N
    return loss_expect


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16,
      'forward_options': {'rtol': 1e-3, 'atol': 1e-3},
      'backward_options': {'rtol': 3e-2, 'atol': 3e-2},
      'double_backward_options': {'rtol': 5e-1, 'atol': 5e-1}},
     {'dtype': numpy.float32,
      'forward_options': {'rtol': 1e-4, 'atol': 1e-4},
      'backward_options': {'rtol': 1e-4, 'atol': 1e-4},
      'double_backward_options': {'rtol': 1e-4, 'atol': 1e-4}},
     {'dtype': numpy.float64,
      'forward_options': {'rtol': 1e-4, 'atol': 1e-4},
      'backward_options': {'rtol': 1e-4, 'atol': 1e-4},
      'double_backward_options': {'rtol': 1e-4, 'atol': 1e-4}},
     ],
    [{'reduce': 'half_squared_sum'},
     {'reduce': 'no'},
     ]
))
class TestCrossCovariance(unittest.TestCase):

    def setUp(self):
        self.y = numpy.random.uniform(-1, 1, (4, 3)).astype(self.dtype)
        self.z = numpy.random.uniform(-1, 1, (4, 2)).astype(self.dtype)
        if self.reduce == 'half_squared_sum':
            gloss_shape = ()
        else:
            gloss_shape = (3, 2)
        self.gloss = numpy.random.uniform(
            -1, 1, gloss_shape).astype(self.dtype)
        self.ggy = numpy.random.uniform(-1, 1, (4, 3)).astype(self.dtype)
        self.ggz = numpy.random.uniform(-1, 1, (4, 2)).astype(self.dtype)

    def check_forward(self, y_data, z_data):
        y = chainer.Variable(y_data)
        z = chainer.Variable(z_data)
        loss = functions.cross_covariance(y, z, self.reduce)

        self.assertEqual(loss.shape, self.gloss.shape)
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        loss_expect = _cross_covariance(y_data, z_data, dtype=self.dtype)
        if self.reduce == 'half_squared_sum':
            loss_expect = numpy.sum(loss_expect ** 2) * 0.5
        numpy.testing.assert_allclose(
            loss_expect, loss_value, **self.forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.y, self.z)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.y), cuda.to_gpu(self.z))

    def check_backward(self, y_data, z_data, gloss_data):
        def f(y, z):
            return functions.cross_covariance(y, z, self.reduce)

        gradient_check.check_backward(
            f, (y_data, z_data), gloss_data, eps=0.02, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.y, self.z, self.gloss)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.y), cuda.to_gpu(self.z),
                            cuda.to_gpu(self.gloss))

    def check_type(self, y_data, z_data, gloss_data):
        y = chainer.Variable(y_data)
        z = chainer.Variable(z_data)
        loss = functions.cross_covariance(y, z, self.reduce)
        loss.grad = gloss_data
        loss.backward()
        self.assertEqual(y_data.dtype, y.grad.dtype)
        self.assertEqual(z_data.dtype, z.grad.dtype)

    def test_backward_type_cpu(self):
        self.check_type(self.y, self.z, self.gloss)

    @attr.gpu
    def test_backward_type_gpu(self):
        self.check_type(cuda.to_gpu(self.y), cuda.to_gpu(self.z),
                        cuda.to_gpu(self.gloss))

    def check_double_backward(self, y_data, z_data, gloss_data, ggy_data,
                              ggz_data):
        def f(y, z):
            return functions.cross_covariance(y, z, self.reduce)

        gradient_check.check_double_backward(
            f, (y_data, z_data), gloss_data, (ggy_data, ggz_data),
            **self.double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.y, self.z, self.gloss, self.ggy, self.ggz)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.y), cuda.to_gpu(self.z), cuda.to_gpu(self.gloss),
            cuda.to_gpu(self.ggy), cuda.to_gpu(self.ggz))


class TestCrossCovarianceInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.y = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.z = numpy.random.uniform(-1, 1, (4, 2)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        y = xp.asarray(self.y)
        z = xp.asarray(self.z)

        with self.assertRaises(ValueError):
            functions.cross_covariance(y, z, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

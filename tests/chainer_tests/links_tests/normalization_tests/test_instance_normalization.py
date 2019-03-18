import unittest

import numpy
import pytest

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*(testing.product({
    'shape': [(2, 3, 5, 5), (5, 3, 15)],
    'test': [True, False],
    'track_avg_stats': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
class TestInstanceNormalization(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        self.link = links.InstanceNormalization(
            3, dtype=self.dtype, track_avg_stats=self.track_avg_stats)
        self.link.cleargrads()
        if self.test and self.track_avg_stats:
            mean = numpy.random.uniform(-1, 1, (3,)).astype(dtype)
            self.link.avg_mean[...] = mean
            var = numpy.random.uniform(0.5, 1, (3,)).astype(dtype)
            self.link.avg_var[...] = var

        self.x = numpy.random.uniform(-1, 1, self.shape).astype(dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(dtype)
        gamma = self.link.gamma.array.copy()
        beta = self.link.beta.array.copy()
        shape = self.shape
        ndim = len(shape)
        expander = (None, Ellipsis) + (None,) * (ndim - 2)

        x_ = self.x.reshape(1, shape[0] * shape[1], *shape[2:])
        if self.test and self.track_avg_stats:
            mean = self.link.avg_mean
            var = self.link.avg_var
            std = numpy.sqrt(var + self.link.eps)
            normalized_x = (
                x_ - numpy.concatenate([mean] * shape[0])[expander]
            ) / numpy.concatenate([std] * shape[0])[expander]
        else:
            aggr_axes = (0,) + tuple(range(2, ndim))
            mean = x_.mean(axis=aggr_axes)
            var = x_.var(axis=aggr_axes)
            std = numpy.sqrt(var + self.link.eps)
            normalized_x = (x_ - mean[expander]) / std[expander]
        y = normalized_x.reshape(shape)
        self.y_expected = gamma[expander] * y + beta[expander]

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with chainer.using_config('train', not self.test):
            y = self.link(x)
        testing.assert_allclose(
            self.y_expected, y.array, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forward_gpu_multi(self):
        with cuda.get_device_from_id(1):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device_from_id(0):
            self.check_forward(x)

    @attr.cudnn
    def test_forward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.test_forward_gpu()

    @attr.multi_gpu(2)
    def test_forward_multi_gpu(self):
        with cuda.get_device_from_id(1):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device_from_id(0):
            self.check_forward(x)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad,
            (self.link.gamma, self.link.beta),
            eps=1e-2, **self.check_backward_options)

    def test_backward_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        self.link.use_cudnn = False
        self.link(numpy.zeros(self.shape, dtype='f'))
        self.test_backward_gpu()


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3]
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.initial_gamma = numpy.random.uniform(-1, 1, self.size)
        self.initial_gamma = self.initial_gamma.astype(numpy.float32)
        self.initial_beta = numpy.random.uniform(-1, 1, self.size)
        self.initial_beta = self.initial_beta.astype(numpy.float32)
        self.link = links.GroupNormalization(self.groups,
                                             initial_gamma=self.initial_gamma,
                                             initial_beta=self.initial_beta)
        self.shape = (1, self.size, 1)

    @condition.retry(3)
    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)

    @attr.gpu
    @condition.retry(3)
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3]
}))
class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.link = links.GroupNormalization(self.groups)
        self.shape = (1, self.size, 1)

    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)

    @attr.gpu
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)


@testing.parameterize(*testing.product({
    'shape': [(2,), (2, 3)],
}))
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.x = numpy.zeros(self.shape, dtype=numpy.float32)
        self.link = links.GroupNormalization(3)

    def test_invalid_shape_cpu(self):
        with self.assertRaises(ValueError):
            self.link(self.x)

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.link.to_gpu()
        with pytest.raises(ValueError):
            self.link(chainer.Variable(cuda.cupy.zeros(self.shape, dtype='f')))


@testing.parameterize(*testing.product({
    'test': [True, False],
    'shape': [(2, 3, 2, 2)],
    'track_avg_stats': [True, False],
}))
class TestInsanceNormalizationWithoutGammaAndBeta(unittest.TestCase):

    def setUp(self):
        self.link = links.InstanceNormalization(
            3, track_avg_stats=self.track_avg_stats,
            use_gamma=False, use_beta=False)
        self.link.cleargrads()
        if self.test and self.track_avg_stats:
            mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
            self.link.avg_mean[...] = mean
            var = numpy.random.uniform(0.5, 1, (3,)).astype(numpy.float32)
            self.link.avg_var[...] = var

        shape = self.shape
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        expander = (None, Ellipsis, None, None)
        x_ = self.x.reshape(1, 6, 2, 2)
        if self.test and self.track_avg_stats:
            mean = self.link.avg_mean
            var = self.link.avg_var
            std = numpy.sqrt(var + self.link.eps)
            y_expected = (x_ - numpy.concatenate([mean] * 2)[expander]) / numpy.concatenate([std] * 2)[expander]
            self.y_expected = y_expected.reshape(shape)
        else:
            aggr_axes = (0, 2, 3)
            mean = x_.mean(axis=aggr_axes)
            var = x_.var(axis=aggr_axes)
            std = numpy.sqrt(var + self.link.eps)
            y_expected = (x_ - mean[expander]) / std[expander]
            self.y_expected = y_expected.reshape(shape)

    def test_no_gamma_and_beta(self):
        assert self.link.gamma is None
        assert self.link.beta is None

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with chainer.using_config('train', not self.test):
            y = self.link(x)
        testing.assert_allclose(self.y_expected, y.array)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        self.check_forward(x)

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device_from_id(0):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device_from_id(1):
            self.check_forward(x)

    @attr.cudnn
    def test_forward_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_forward_gpu()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self.link, x_data, y_grad,
                                      eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, gy)

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_backward_gpu()


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3]
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.initial_gamma = numpy.random.uniform(-1, 1, self.size)
        self.initial_gamma = self.initial_gamma.astype(numpy.float32)
        self.initial_beta = numpy.random.uniform(-1, 1, self.size)
        self.initial_beta = self.initial_beta.astype(numpy.float32)
        self.link = links.GroupNormalization(self.groups,
                                             initial_gamma=self.initial_gamma,
                                             initial_beta=self.initial_beta)
        self.shape = (1, self.size, 1)

    @condition.retry(3)
    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)

    @attr.gpu
    @condition.retry(3)
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3]
}))
class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.link = links.InstanceNormalization(self.groups)
        self.shape = (1, self.size, 1)

    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)

    @attr.gpu
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)


@testing.parameterize(*testing.product({
    'shape': [(2,), (2, 3)],
}))
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.x = numpy.zeros(self.shape, dtype=numpy.float32)
        self.link = links.InstanceNormalization(3)

    def test_invalid_shape_cpu(self):
        with self.assertRaises(ValueError):
            self.link(self.x)

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.link.to_gpu()
        with pytest.raises(ValueError):
            self.link(chainer.Variable(cuda.cupy.zeros(self.shape, dtype='f')))


@testing.parameterize(*testing.product({
    'test': [True, False],
    'shape': [(2, 3, 2, 2)],
    'track_avg_stats': [True, False],
}))
class TestInsanceNormalizationWithoutGammaAndBeta(unittest.TestCase):

    def setUp(self):
        self.link = links.InstanceNormalization(
            3, track_avg_stats=self.track_avg_stats,
            use_gamma=False, use_beta=False)
        self.link.cleargrads()
        if self.test and self.track_avg_stats:
            mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
            self.link.avg_mean[...] = mean
            var = numpy.random.uniform(0.5, 1, (3,)).astype(numpy.float32)
            self.link.avg_var[...] = var

        shape = self.shape
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        expander = (None, Ellipsis, None, None)
        x_ = self.x.reshape(1, 6, 2, 2)
        if self.test and self.track_avg_stats:
            mean = self.link.avg_mean
            var = self.link.avg_var
            std = numpy.sqrt(var + self.link.eps)
            y_expected = (
                x_ - numpy.concatenate([mean] * 2)[expander]
            ) / numpy.concatenate([std] * 2)[expander]
            self.y_expected = y_expected.reshape(shape)
        else:
            aggr_axes = (0, 2, 3)
            mean = x_.mean(axis=aggr_axes)
            var = x_.var(axis=aggr_axes)
            std = numpy.sqrt(var + self.link.eps)
            y_expected = (x_ - mean[expander]) / std[expander]
            self.y_expected = y_expected.reshape(shape)

    def test_no_gamma_and_beta(self):
        assert self.link.gamma is None
        assert self.link.beta is None

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with chainer.using_config('train', not self.test):
            y = self.link(x)
        testing.assert_allclose(self.y_expected, y.array)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        self.check_forward(x)

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device_from_id(0):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device_from_id(1):
            self.check_forward(x)

    @attr.cudnn
    def test_forward_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_forward_gpu()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self.link, x_data, y_grad,
                                      eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, gy)

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_backward_gpu()


testing.run_module(__name__, __file__)

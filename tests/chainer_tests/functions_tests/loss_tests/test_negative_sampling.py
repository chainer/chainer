import unittest

import numpy
import pytest
import six

import chainer
from chainer.backend import CpuDevice
from chainer.backends import cuda
from chainer import functions
from chainer.functions.loss import negative_sampling
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def make_sampler(backend_config, high):
    # To fix samples, use fixed samples.
    def sampler(shape):
        s = numpy.arange(numpy.prod(shape)) % high
        s = s.reshape(shape).astype(numpy.int32)
        return backend_config.get_array(s)
    return sampler


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    't': [[0, 2], [-1, 1, 2]],
    'reduce': ['sum', 'no'],
}))
@testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestNegativeSamplingFunction(unittest.TestCase):

    in_size = 3
    sample_size = 2
    label_size = 5

    def setUp(self):

        batch = len(self.t)
        x_shape = (batch, self.in_size)
        w_shape = (self.label_size, self.in_size)

        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.t = numpy.array(self.t).astype(numpy.int32)
        self.w = numpy.random.uniform(-1, 1, w_shape).astype(self.dtype)

        if self.reduce == 'no':
            g_shape = self.t.shape
        elif self.reduce == 'sum':
            g_shape = ()

        self.gy = numpy.random.uniform(-1, 1, g_shape).astype(self.dtype)

        self.ggx = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        self.ggw = numpy.random.uniform(-1, 1, w_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'eps': 1e-2, 'atol': 5e-4, 'rtol': 5e-3}
        self.check_double_backward_options = {
            'eps': 1e-2, 'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options['dtype'] = numpy.float64
            self.check_double_backward_options['dtype'] = numpy.float64

    def test_forward(self, backend_config):
        sampler = make_sampler(backend_config, self.label_size)
        x_data = backend_config.get_array(self.x)
        t_data = backend_config.get_array(self.t)
        w_data = backend_config.get_array(self.w)
        batch_size = len(self.t)
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data, requires_grad=False)
        w = chainer.Variable(w_data)

        # return_samples=False
        y = functions.negative_sampling(
            x, t, w, sampler, self.sample_size, reduce=self.reduce)
        assert y.dtype == self.dtype

        # return_samples=True
        y_, samples = functions.negative_sampling(
            x, t, w, sampler, self.sample_size, reduce=self.reduce,
            return_samples=True)

        xp = chainer.backend.get_array_module(x)
        assert isinstance(samples, xp.ndarray)
        assert samples.dtype == numpy.int32
        assert samples.shape == (batch_size, self.sample_size + 1)

        # Sampler is deterministic, so y and y_ should equal.
        assert y.dtype == y_.dtype
        cpu_device = CpuDevice()
        numpy.testing.assert_array_equal(
            cpu_device.send(y.array), cpu_device.send(y_.array))

        assert y.shape == self.gy.shape

        samples = cpu_device.send(samples)

        loss = numpy.empty((len(self.x),), self.dtype)
        for i in six.moves.range(len(self.x)):
            ix = self.x[i]
            it = self.t[i]
            if it == -1:
                loss[i] = 0
            else:
                iw = self.w[samples[i]]

                f = iw.dot(ix)
                # first one is positive example
                f[0] *= -1
                loss[i] = numpy.logaddexp(f, 0).sum()

        if self.reduce == 'sum':
            loss = loss.sum()

        assert y.dtype == loss.dtype
        testing.assert_allclose(y.data, loss, **self.check_forward_options)

    def test_backward(self, backend_config):
        sampler = make_sampler(backend_config, self.label_size)
        x_data = backend_config.get_array(self.x)
        t_data = backend_config.get_array(self.t)
        w_data = backend_config.get_array(self.w)
        y_grad = backend_config.get_array(self.gy)

        def f(x, w):
            return functions.negative_sampling(
                x, t_data, w, sampler, self.sample_size, reduce=self.reduce)

        with backend_config:
            gradient_check.check_backward(
                f, (x_data, w_data), y_grad, **self.check_backward_options)

    def test_double_backward(self, backend_config):
        sampler = make_sampler(backend_config, self.label_size)
        x_data = backend_config.get_array(self.x)
        t_data = backend_config.get_array(self.t)
        w_data = backend_config.get_array(self.w)
        y_grad = backend_config.get_array(self.gy)
        x_grad_grad = backend_config.get_array(self.ggx)
        w_grad_grad = backend_config.get_array(self.ggw)

        def f(x, w):
            return functions.negative_sampling(
                x, t_data, w, sampler, self.sample_size, reduce=self.reduce)

        with backend_config:
            gradient_check.check_double_backward(
                f, (x_data, w_data), y_grad, (x_grad_grad, w_grad_grad),
                **self.check_double_backward_options)


class TestNegativeSamplingInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 2, (2,)).astype(numpy.int32)
        self.w = numpy.random.uniform(-1, 1, (5, 3)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        w = xp.asarray(self.w)

        with pytest.raises(ValueError):
            negative_sampling.negative_sampling(
                x, t, w, make_sampler(xp, 5), 2, reduce='invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

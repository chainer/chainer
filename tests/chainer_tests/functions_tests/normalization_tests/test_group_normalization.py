import six
import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _simple_group_normalization(x, groups, gamma, beta, xp, eps=1e-5):
    batch_size, channels = x.shape[:2]
    x_reshape = x.reshape(batch_size, groups, channels // groups, -1)

    mu = xp.mean(x_reshape, axis=(2, 3), keepdims=True)
    sigma = xp.std(x_reshape, axis=(2, 3), keepdims=True)

    x_reshape = (x_reshape - mu) / (sigma + eps)
    x = x_reshape.reshape(x.shape)

    for i in six.moves.xrange(x.ndim):
        if i != 1:  # except for channel dim
            gamma = xp.expand_dims(gamma, i)
            beta = xp.expand_dims(beta, i)

    return x * gamma + beta


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'groups': [1, 2, 4],
    'dtype': [numpy.float32],
})))
class TestGroupNormalization(unittest.TestCase):

    def setUp(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gamma = numpy.random.uniform(-1, 1, self.shape[1]).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, self.shape[1]).astype(self.dtype)
        self.args = [x, gamma, beta]
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = [numpy.random.uniform(-1, 1, arg.shape).astype(arg.dtype)
                    for arg in self.args]

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, args):
        xp = backend.get_array_module(*args)

        def func(*args_):
            return functions.group_normalization(
                *[args_[0], self.groups, args_[1], args_[2]])

        y = func(*args)
        self.assertEqual(y.data.dtype, self.dtype)

        # Verify that implementation using batch normalization is correct
        y_simple_gn = _simple_group_normalization(
            args[0], self.groups, args[1], args[2], xp)
        testing.assert_allclose(
            y.data, y_simple_gn, **self.check_forward_options)

        # Verify that forward isn't be affected by batch size
        if self.shape[0] > 1:
            y_one_each = chainer.functions.concat(
                [func(*[xp.expand_dims(one_x, axis=0), args[1], args[2]])
                 for one_x in args[0]], axis=0)
            testing.assert_allclose(
                y.data, y_one_each.data, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(arg) for arg in self.args])

    def check_backward(self, args, y_grad):
        def func(*args_):
            return functions.group_normalization(
                *[args_[0], self.groups, args_[1], args_[2]])

        gradient_check.check_backward(
            func, args, y_grad,
            eps=1e-2, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(arg) for arg in self.args], cuda.to_gpu(self.gy))

    def check_double_backward(self, args, y_grad, x_grad_grad):
        def func(*args_):
            return functions.group_normalization(
                *[args_[0], self.groups, args_[1], args_[2]])

        gradient_check.check_double_backward(
            func, args, y_grad, x_grad_grad,
            eps=1e-2, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.args, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            [cuda.to_gpu(arg) for arg in self.args],
            cuda.to_gpu(self.gy), [cuda.to_gpu(ggx_) for ggx_ in self.ggx])


testing.run_module(__name__, __file__)

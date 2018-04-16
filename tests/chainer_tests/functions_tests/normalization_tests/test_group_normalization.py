import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 2, 2), (5, 4, 2)],
    'n_groups': [1, 2, 4],
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
        def func(*args_):
            return functions.group_normalization(
                *[args_[0], self.n_groups, args_[1], args_[2]])

        y = func(*args)
        self.assertEqual(y.data.dtype, self.dtype)

        # Verify that forward isn't be affected by batch size
        if self.shape[0] > 1:
            xp = cuda.get_array_module(*args)
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
                *[args_[0], self.n_groups, args_[1], args_[2]])

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
            y = functions.group_normalization(
                *[args_[0], self.n_groups, args_[1], args_[2]])
            return y * y

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

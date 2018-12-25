import unittest

import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*(testing.product({
    'batchsize': [1, 5],
    'size': [10, 20],
    'dtype': [numpy.float32],
    'eps': [1e-5, 1e-1],
})))
class TestLayerNormalization(unittest.TestCase):

    def setUp(self):
        shape = self.batchsize, self.size
        size = numpy.prod(shape) // shape[0]
        x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        gamma = numpy.random.uniform(-1, 1, size).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, size).astype(self.dtype)
        self.args = (x, gamma, beta)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.ggx = [numpy.random.uniform(-1, 1, _.shape).astype(_.dtype)
                    for _ in self.args]

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

        mean = numpy.mean(x, axis=1, keepdims=True)
        var = numpy.mean(numpy.square(x - mean), axis=1, keepdims=True)
        std = numpy.sqrt(var + self.eps)
        self.y_expected = (
            numpy.expand_dims(gamma, axis=0) * (x - mean) / std
            + numpy.expand_dims(beta, axis=0))

    def check_forward(self, args):
        x_data = args[0]

        def func(x):
            args_ = x, args[1], args[2]
            return functions.layer_normalization(*args_, eps=self.eps)

        y = func(x_data)
        self.assertEqual(y.data.dtype, self.dtype)

        testing.assert_allclose(
            y.data, self.y_expected, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(_) for _ in self.args])

    def check_backward(self, args, y_grad):
        def func(*args_):
            return functions.layer_normalization(*args_, eps=self.eps)

        gradient_check.check_backward(
            func, args, y_grad,
            eps=1e-2, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.args, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(_) for _ in self.args],
            cuda.to_gpu(self.gy))

    def check_double_backward(self, args, y_grad, x_grad_grad):
        def func(*args_):
            return functions.layer_normalization(*args_, eps=self.eps)

        gradient_check.check_double_backward(
            func, args, y_grad, x_grad_grad,
            eps=1e-2, **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.args, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            [cuda.to_gpu(_) for _ in self.args],
            cuda.to_gpu(self.gy), [cuda.to_gpu(_) for _ in self.ggx])


testing.run_module(__name__, __file__)

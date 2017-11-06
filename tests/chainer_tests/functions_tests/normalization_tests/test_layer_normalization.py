import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _batch_normalization(expander, gamma, beta, x, mean, var):
    mean = mean[expander]
    std = numpy.sqrt(var)[expander]
    y_expect = (gamma[expander] * (x - mean) / std + beta[expander])
    return y_expect


@testing.parameterize(*(testing.product({
    'batchsize': [1, 5],
    'size': [10, 20],
    'dtype': [numpy.float32],
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

    def check_forward(self, args):
        x_data = args[0]

        def func(x):
            args_ = x, args[1], args[2]
            return functions.layer_normalization(*args_)

        y = func(x_data)
        self.assertEqual(y.data.dtype, self.dtype)

        unbatched_concat_y = chainer.functions.concat(
            [func(one_x[None, ]) for one_x in x_data], axis=0)

        testing.assert_allclose(
            y.data, unbatched_concat_y.data, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.args)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(_) for _ in self.args])

    def check_backward(self, args, y_grad):
        def func(*args_):
            return functions.layer_normalization(*args_)

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
            y = functions.layer_normalization(*args_)
            return y * y

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

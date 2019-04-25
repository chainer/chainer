import unittest

import numpy

import chainer
from chainer import functions
from chainer import gradient_check
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (1,), (1, 2, 3, 4, 5, 6)],
    'Wdim': [0, 1, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
@chainer.testing.backend.inject_backend_tests(
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
class TestPReLU(unittest.TestCase):

    def setUp(self):
        # Avoid unstability of numerical grad
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.x[(-0.05 < self.x) & (self.x < 0.05)] = 0.5
        self.W = numpy.random.uniform(
            -1, 1, self.shape[1:1 + self.Wdim]).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggW = numpy.random.uniform(
            -1, 1, self.W.shape).astype(self.dtype)

        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-3})

        self.check_double_backward_options = {
            'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}
        if self.dtype == numpy.float16:
            self.check_double_backward_options.update(
                {'atol': 5e-3, 'rtol': 5e-2})

    def test_forward(self, backend_config):
        x_data, W_data = backend_config.get_array((self.x, self.W))
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        y = functions.prelu(x, W)
        self.assertEqual(y.data.dtype, self.dtype)

        y_expect = self.x.copy()
        masked = numpy.ma.masked_greater_equal(y_expect, 0, copy=False)
        shape = (1,) + W.shape + (1,) * (x.ndim - W.ndim - 1)
        masked *= self.W.reshape(shape)
        testing.assert_allclose(y_expect, y.data)

    def test_backward(self, backend_config):
        x_data, W_data, y_grad = (
            backend_config.get_array((self.x, self.W, self.gy)))

        with backend_config:
            gradient_check.check_backward(
                functions.prelu, (x_data, W_data), y_grad,
                **self.check_backward_options)

    def test_double_backward(self, backend_config):
        x_data, W_data = backend_config.get_array((self.x, self.W))
        y_grad = backend_config.get_array(self.gy)
        x_grad_grad, W_grad_grad = (
            backend_config.get_array((self.ggx, self.ggW)))

        with backend_config:
            gradient_check.check_double_backward(
                functions.prelu, (x_data, W_data), y_grad,
                (x_grad_grad, W_grad_grad),
                **self.check_double_backward_options)


testing.run_module(__name__, __file__)

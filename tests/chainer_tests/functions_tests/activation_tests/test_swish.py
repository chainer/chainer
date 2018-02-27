import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'x_shape': (4, 3, 2), 'beta_shape': (3,),
         'extended_beta_shape': (1, 3, 1)},
        {'x_shape': (4, 3, 2), 'beta_shape': (3, 2),
         'extended_beta_shape': (1, 3, 2)},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
@testing.fix_random()
class TestSwish(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.beta = numpy.random.uniform(-1, 1, self.beta_shape)\
            .astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.ggb = numpy.random.uniform(-1, 1, self.beta_shape)\
            .astype(self.dtype)

        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, beta_data):
        x = chainer.Variable(x_data)
        beta = chainer.Variable(beta_data)
        y = functions.swish(x, beta)
        self.assertEqual(y.data.dtype, self.dtype)

        beta = chainer.functions.broadcast_to(beta.reshape(
            self.extended_beta_shape), x.shape)
        y_expect = x * chainer.functions.sigmoid(beta * x)
        testing.assert_allclose(y_expect.data, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.beta)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.beta))

    def check_backward(self, x_data, beta_data, gy_data):
        gradient_check.check_backward(
            functions.swish, (x_data, beta_data), gy_data,
            dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.beta, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.beta),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, beta_data, gy_data,
                              ggx_data, ggb_data):
        gradient_check.check_double_backward(
            functions.swish, (x_data, beta_data), gy_data,
            (ggx_data, ggb_data), dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.beta, self.gy,
                                   self.ggx, self.ggb)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.beta),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            cuda.to_gpu(self.ggb))


testing.run_module(__name__, __file__)

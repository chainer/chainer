import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _dropout(x, creator):
    return x * creator.mask


@testing.parameterize(
    {'dtype': numpy.float16, 'ratio': 0.1},
    {'dtype': numpy.float32, 'ratio': 0.3},
    {'dtype': numpy.float64, 'ratio': 0.5},
    {'dtype': numpy.float64, 'ratio': 0.0},
)
class TestDropout(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)

        self.check_backward_options = {'dtype': 'd'}
        self.check_double_backward_options = {'dtype': 'd'}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {
                'dtype': 'd', 'atol': 5e-4, 'rtol': 5e-3}

    def check_type_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.dropout(x)
        self.assertEqual(y.data.dtype, self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.dropout(x, self.ratio)
        if self.ratio == 0.0:
            y_expect = x_data
        else:
            y_expect = _dropout(x_data, y.creator_node)
        testing.assert_allclose(y_expect, y.data)

    def check_backward(self, x_data, y_grad):
        dropout = functions.Dropout(self.ratio)

        def f(x):
            return dropout.apply((x,))[0]

        gradient_check.check_backward(
            f, x_data, y_grad,
            **self.check_double_backward_options)

    def test_type_forward_cpu(self):
        self.check_type_forward(self.x)

    @attr.gpu
    def test_type_forward_gpu(self):
        self.check_type_forward(cuda.to_gpu(self.x))

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        dropout = functions.Dropout(self.ratio)

        def f(x):
            x, = dropout.apply((x,))
            return x * x

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x),
                                   cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))

    def check_immutable(self, x_data):
        d = functions.Dropout(0.5)
        y1, = d.apply((chainer.Variable(x_data),))
        y2, = d.apply((chainer.Variable(x_data),))
        testing.assert_allclose(y1.data, y2.data)

    def test_immutable_cpu(self):
        self.check_immutable(self.x)

    @attr.gpu
    def test_immutable_gpu(self):
        self.check_immutable(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)

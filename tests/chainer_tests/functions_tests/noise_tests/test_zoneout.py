import unittest

import mock
import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _zoneout(h, x, creator):
    h_next = h * creator.flag_h + x * creator.flag_x
    return h_next


@testing.parameterize(
    {'ratio': 1},
    {'ratio': 0},
    {'ratio': 0.5},
    {'ratio': 0.25},
)
class TestZoneout(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggh = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)

    def check_forward(self, h_data, x_data):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        h_next = functions.zoneout(h, x, self.ratio)
        if self.ratio == 0:
            h_next_expect = x_data
        elif self.ratio == 1:
            h_next_expect = h_data
        else:
            h_next_expect = _zoneout(h_data, x_data, h_next.creator)
        testing.assert_allclose(h_next.data, h_next_expect)

    def check_backward(self, h_data, x_data, y_grad):
        h = chainer.Variable(h_data)
        x = chainer.Variable(x_data)
        y = functions.zoneout(h, x, self.ratio)
        d = {'creator': y.creator}
        y.grad = y_grad
        y.backward()

        def f():
            creator = d['creator']
            y = _zoneout(h_data, x_data, creator)
            return y,
        gh, gx, = gradient_check.numerical_grad(f, (h.data, x.data,),
                                                (y.grad,))
        testing.assert_allclose(gh, h.grad, atol=1e-3)
        testing.assert_allclose(gx, x.grad, atol=1e-3)

    def check_double_backward(
            self, h_data, x_data, y_grad, h_grad_grad, x_grad_grad):
        xp = backend.get_array_module(h_data)
        flag_x = xp.random.rand(*x_data.shape)

        def f(h, x):
            # As forward computation is executed multiple times in
            # check_double_backward, use a fixed flag.
            xp_str = 'numpy' if xp is numpy else 'cupy'
            with mock.patch(
                    '{}.random.rand'.format(xp_str),
                    return_value=flag_x) as mock_rand:
                y = functions.zoneout(h, x, self.ratio)
                mock_rand.assert_called_once_with(*x.shape)
            return y

        gradient_check.check_double_backward(
            f, (h_data, x_data), y_grad, (h_grad_grad, x_grad_grad),
            dtype=numpy.float64, atol=1e-7, rtol=1e-7)

    def test_forward_cpu(self):
        self.check_forward(self.h, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.h), cuda.to_gpu(self.x))

    def test_backward_cpu(self):
        self.check_backward(self.h, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.h),
                            cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

    def test_double_backward_cpu(self):
        self.check_double_backward(self.h, self.x, self.gy, self.ggh, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.h),
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggh),
            cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)

import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 4), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
@testing.fix_random()
class TestHardSigmoid(unittest.TestCase):

    def setUp(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Avoid unstability of numerical grad
        x[((-0.35 < x) & (x < -0.15)) | ((0.15 < x) & (x < 0.35))] = 0.5
        self.x = x

        self.check_forward_option = {}
        self.check_backward_option = {}
        if self.dtype is numpy.float16:
            self.check_forward_option = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_option = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.hard_sigmoid(x)
        self.assertIs(y.data.dtype, x_data.dtype)
        expect = (self.x * 0.2 + 0.5).clip(0, 1)
        testing.assert_allclose(
            y.data, expect, **self.check_forward_option)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, grad):
        gradient_check.check_backward(
            functions.HardSigmoid(), x_data, grad, dtype=numpy.float64,
            **self.check_backward_option)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)

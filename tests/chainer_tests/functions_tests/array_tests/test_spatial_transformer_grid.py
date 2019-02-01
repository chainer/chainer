import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_cudnn': ['always', 'never'],
}))
class TestSpatialTransformerGrid(unittest.TestCase):

    def setUp(self):
        B = 3
        self.theta = numpy.random.uniform(size=(B, 2, 3)).astype(self.dtype)
        self.output_shape = (5, 6)
        self.grads = numpy.random.uniform(
            size=(B, 2) + self.output_shape).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-1}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {}

    def check_forward(self, theta, output_shape):
        grid = functions.spatial_transformer_grid(theta, output_shape).data

        theta = cuda.to_cpu(theta)
        B = theta.shape[0]
        H, W = output_shape

        expected = []
        for b in range(B):
            for i in numpy.linspace(-1., 1., H):
                for j in numpy.linspace(-1., 1., W):
                    coord = numpy.array([j, i, 1])
                    expected.append(self.theta[b].dot(coord))
        expected = numpy.array(
            expected).reshape(B, H, W, 2).transpose(0, 3, 1, 2)
        testing.assert_allclose(grid, expected, **self.check_forward_options)
        self.assertEqual(grid.dtype, self.dtype)

    def test_forward_cpu(self):
        self.check_forward(self.theta, self.output_shape)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.theta), self.output_shape)

    def check_backward(self, theta, output_shape, grads):
        def f(theta):
            return functions.spatial_transformer_grid(theta, output_shape)

        with chainer.using_config('use_cudnn', self.use_cudnn):
            gradient_check.check_backward(
                f, (theta,), (grads,), dtype=numpy.float64,
                **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.theta, self.output_shape, self.grads)

    @attr.gpu
    def test_backward_gpu(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.check_backward(cuda.to_gpu(self.theta), self.output_shape,
                                cuda.to_gpu(self.grads))


testing.run_module(__name__, __file__)

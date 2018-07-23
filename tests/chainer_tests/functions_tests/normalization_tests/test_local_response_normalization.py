import unittest

import numpy
import six

from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import backend


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + [{'use_cuda': True}])
class TestLocalResponseNormalization(unittest.TestCase):

    def setUp(self):
        x = numpy.random.uniform(-1, 1, (2, 7, 3, 2)).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, (2, 7, 3, 2)).astype(self.dtype)

        self.inputs = [x]
        self.grad_outputs = [gy]

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-3}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {'atol': 3e-4, 'rtol': 3e-3}

    def forward_cpu(self, inputs):
        # Naive implementation
        x, = inputs
        y_expect = numpy.zeros_like(x)
        for n, c, h, w in numpy.ndindex(x.shape):
            s = 0
            for i in six.moves.range(max(0, c - 2), min(7, c + 2)):
                s += x[n, i, h, w] ** 2
            denom = (2 + 1e-4 * s) ** .75
            y_expect[n, c, h, w] = x[n, c, h, w] / denom
        return y_expect,

    def check_forward(self, inputs, backend_config):
        y_expect, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)

        with backend_config:
            y = functions.local_response_normalization(*inputs)

        assert y.data.dtype == self.dtype
        testing.assert_allclose(y_expect, y.data, **self.check_forward_options)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)

        with backend_config:
            gradient_check.check_backward(
                functions.local_response_normalization, inputs, grad_outputs,
                eps=1, dtype=numpy.float64, **self.check_backward_options)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)


testing.run_module(__name__, __file__)

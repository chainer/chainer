import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import force_array


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'x_min_max': [
        (-0.75, 1.53),
        (numpy.float32(-0.75), numpy.float32(1.53)),
        (-1, 2),
    ]
}))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestClip(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-3, 3, self.shape).astype(self.dtype)
        # Avoid values around x_min and x_max for stability of numerical
        # gradient
        x_min, x_max = self.x_min_max
        x_min = float(x_min)
        x_max = float(x_max)
        eps = 0.01
        for ind in numpy.ndindex(x.shape):
            if x_min - eps < x[ind] < x_min + eps:
                x[ind] = -0.5
            elif x_max - eps < x[ind] < x_max + eps:
                x[ind] = 0.5
        return x,

    def forward(self, inputs, device):
        x, = inputs
        x_min, x_max = self.x_min_max
        y = functions.clip(x, x_min, x_max)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        x_min, x_max = self.x_min_max
        expected = numpy.clip(x, x_min, x_max)
        expected = force_array(expected)
        return expected,


class TestClipInvalidInterval(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            functions.clip(self.x, 1.0, -1.0)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestClipBorderGrad(unittest.TestCase):

    def setUp(self):
        self.x = numpy.arange(6, dtype=self.dtype)
        self.x_min = 1.0
        self.x_max = 4.0
        self.expected = numpy.asarray([0, 1, 1, 1, 1, 0], dtype=self.dtype)

    def check_border_grad(self, x, expected):
        x = chainer.Variable(x)
        y = functions.clip(x, self.x_min, self.x_max)
        l = functions.sum(y)
        l.backward()
        testing.assert_allclose(x.grad, expected, atol=0, rtol=0)

    def test_border_grad_cpu(self):
        self.check_border_grad(self.x, self.expected)

    @attr.gpu
    def test_border_grad_gpu(self):
        self.check_border_grad(cuda.to_gpu(self.x), cuda.to_gpu(self.expected))


testing.run_module(__name__, __file__)

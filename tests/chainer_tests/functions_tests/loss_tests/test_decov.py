import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


def _decov(h):
    h_mean = h.mean(axis=0)
    N, M = h.shape
    loss_expect = numpy.zeros((M, M), dtype=h.dtype)
    for i in six.moves.range(M):
        for j in six.moves.range(M):
            if i != j:
                for n in six.moves.range(N):
                    loss_expect[i, j] += (h[n, i] - h_mean[i]) * (
                        h[n, j] - h_mean[j])
    return loss_expect / N


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['half_squared_sum', 'no'],
}))
@backend.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + [{
        'use_cuda': True,
    }]
)
class TestDeCov(testing.FunctionTestCase):

    skip_double_backward_test = True

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({'atol': 3e-2, 'eps': 0.02})
        else:
            self.check_forward_options.update({'rtol': 1e-4, 'atol': 1e-4})
            self.check_backward_options.update({'atol': 1e-3, 'eps': 0.02})

    def generate_inputs(self):
        h = numpy.random.uniform(-1, 1, (4, 3)).astype(self.dtype)
        return h,

    def forward_expected(self, inputs):
        h, = inputs
        loss_expect = _decov(h)
        if self.reduce == 'half_squared_sum':
            loss_expect = (loss_expect ** 2).sum() * 0.5
        return chainer.utils.force_array(loss_expect, self.dtype),

    def forward(self, inputs, device):
        h, = inputs
        loss = functions.decov(h, self.reduce)
        return loss,


class TestDeconvInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.h = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        h = xp.asarray(self.h)

        with self.assertRaises(ValueError):
            functions.decov(h, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

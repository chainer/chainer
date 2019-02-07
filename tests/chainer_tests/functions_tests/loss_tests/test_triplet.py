import unittest

import numpy
import six

from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer import utils
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batchsize': [5, 10],
    'input_dim': [2, 3],
    'margin': [0.1, 0.5],
    'reduce': ['mean', 'no']
}))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {}
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
class TestTriplet(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):

        if self.dtype == numpy.float16:
            self.check_forward_options = {'rtol': 5e-3, 'atol': 5e-3}
            self.check_backward_options = {'rtol': 5e-2, 'atol': 5e-2}
            self.check_double_backward_options = {'rtol': 1e-3, 'atol': 1e-3}
        elif self.dtype == numpy.float32:
            self.check_forward_options = {'rtol': 1e-4, 'atol': 1e-4}
            self.check_backward_options = {'rtol': 5e-4, 'atol': 5e-4}
            self.check_double_backward_options = {'rtol': 1e-3, 'atol': 1e-3}
        elif self.dtype == numpy.float64:
            self.check_forward_options = {'rtol': 1e-4, 'atol': 1e-4}
            self.check_backward_options = {'rtol': 5e-4, 'atol': 5e-4}
            self.check_double_backward_options = {'rtol': 1e-3, 'atol': 1e-3}
        else:
            raise ValueError('invalid dtype')

    def generate_inputs(self):
        eps = 1e-3
        x_shape = (self.batchsize, self.input_dim)

        # Sample differentiable inputs
        while True:
            a = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            p = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            n = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
            if (abs(a - p) < 2 * eps).any():
                continue
            if (abs(a - n) < 2 * eps).any():
                continue
            dist = numpy.sum((a - p) ** 2 - (a - n) ** 2, axis=1) + self.margin
            if (abs(dist) < 2 * eps).any():
                continue
            break
        return a, p, n

    def forward(self, inputs, device):
        anchor, positive, negative = inputs
        return functions.triplet(
            anchor, positive, negative, self.margin, self.reduce),

    def forward_expected(self, inputs):
        anchor, positive, negative = inputs
        loss_expect = numpy.empty((anchor.shape[0],), dtype=self.dtype)
        for i in six.moves.range(anchor.shape[0]):
            ad, pd, nd = anchor[i], positive[i], negative[i]
            dp = numpy.sum((ad - pd) ** 2)
            dn = numpy.sum((ad - nd) ** 2)
            loss_expect[i] = max((dp - dn + self.margin), 0)
        if self.reduce == 'mean':
            loss_expect = loss_expect.mean()

        return utils.force_array(loss_expect, self.dtype),


class TestContrastiveInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.a = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.p = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.n = numpy.random.randint(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        a = xp.asarray(self.a)
        p = xp.asarray(self.p)
        n = xp.asarray(self.n)

        with self.assertRaises(ValueError):
            functions.triplet(a, p, n, reduce='invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

import unittest

import numpy
import pytest
import six

from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'batchsize': [5, 10],
    'input_dim': [2, 3],
    'margin': [0.1, 0.5],
    'reduce': ['mean', 'no']
}))
class TestTriplet(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 5e-2, 'atol': 5e-2})
            self.check_double_backward_options.update({
                'rtol': 1e-3, 'atol': 1e-3})

    def generate_inputs(self):
        x_shape = (self.batchsize, self.input_dim)
        a = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        p = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        n = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        return a, p, n

    def forward(self, inputs, device):
        a, p, n = inputs
        loss = functions.triplet(a, p, n, self.margin, self.reduce)
        return loss,

    def forward_expected(self, inputs):
        a, p, n = inputs
        loss_expect = numpy.empty((a.shape[0],), dtype=self.dtype)
        for i in six.moves.range(a.shape[0]):
            ad, pd, nd = a[i], p[i], n[i]
            dp = numpy.sum((ad - pd) ** 2)
            dn = numpy.sum((ad - nd) ** 2)
            loss_expect[i] = max((dp - dn + self.margin), 0)
        if self.reduce == 'mean':
            loss_expect = numpy.asarray(loss_expect.mean())
        return loss_expect,


class TestTripletNegativeMargin(unittest.TestCase):

    def test_negative_margin(self):
        x_shape = (5, 2)
        dtype = numpy.float32
        a = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        p = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        n = numpy.random.uniform(-1, 1, x_shape).astype(dtype)

        with pytest.raises(ValueError):
            functions.triplet(a, p, n, -1, 'sum')


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

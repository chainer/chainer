import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


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
)
@testing.parameterize(*testing.product({
    'train': [True, False],
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRReLU(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        # Assumption l < u
        self.l = numpy.random.uniform(0, 1)
        self.u = numpy.random.uniform(0, 1)
        if self.l >= self.u:
            self.l, self.u = self.u, self.l

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

        # cast self.r later because check_backward casts only x
        self.r = numpy.random.uniform(self.l, self.u, self.shape)

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        r = self.r.astype(x.dtype)
        r = device.send(r)
        with chainer.using_config('train', self.train):
            y = functions.rrelu(x, l=self.l, u=self.u, r=r)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        r = self.r.astype(self.dtype)
        if self.train:
            expected = numpy.where(x >= 0, x, x * r)
        else:
            r_test = numpy.mean([self.l, self.u]).astype(self.dtype)
            expected = numpy.where(x >= 0, x, x * r_test)
        return expected,


@testing.parameterize(*testing.product({
    'specify_r': [True, False],
    'return_r': [True, False],
    'train': [True, False],
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestRReLUR(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Assumption l < u
        self.l = numpy.random.uniform(0, 1)
        self.u = numpy.random.uniform(0, 1)
        if self.l >= self.u:
            self.l, self.u = self.u, self.l
        self.r = numpy.random.uniform(
            self.l, self.u, self.x.shape).astype(self.x.dtype)

    def _check(self):
        r = self.r if self.specify_r else None
        return_r = self.return_r
        with chainer.using_config('train', self.train):
            out = functions.rrelu(
                self.x, self.l, self.u, r=r, return_r=return_r)

        if not return_r:
            return

        out, out_r = out
        assert isinstance(out_r, type(out.array))
        if r is None:
            assert out_r.shape == out.array.shape
        else:
            if self.train:
                assert out_r is r

    def test_cpu(self):
        with chainer.using_config('use_ideep', 'never'):
            self._check()

    @attr.gpu
    def test_gpu(self):
        self.x = cuda.to_gpu(self.x)
        self.r = cuda.to_gpu(self.r)
        self._check()


testing.run_module(__name__, __file__)

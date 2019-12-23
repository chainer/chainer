import numpy

import chainer.functions as F
from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'shape': [(5, 5), (1, 1)]
}))
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
    # TODO(niboshi): Add ChainerX tests
)
class TestCholesky(testing.FunctionTestCase):

    def random_matrix(self, shape, dtype, scale, sym=False):
        m, n = shape[-2:]
        dtype = numpy.dtype(dtype)
        assert dtype.kind in 'iufc'
        low_s, high_s = scale
        bias = None
        if dtype.kind in 'iu':
            err = numpy.sqrt(m * n) / 2.
            low_s += err
            high_s -= err
            if dtype.kind in 'u':
                assert sym, (
                    'generating nonsymmetric matrix with uint cells is not'
                    ' supported')
                # (singular value of numpy.ones((m, n))) <= \sqrt{mn}
                high_s = bias = high_s / (1 + numpy.sqrt(m * n))
        assert low_s <= high_s
        a = numpy.random.standard_normal(shape)
        u, s, vh = numpy.linalg.svd(a)
        new_s = numpy.random.uniform(low_s, high_s, s.shape)
        if sym:
            assert m == n
            new_a = numpy.einsum('...ij,...j,...kj', u, new_s, u)
        else:
            new_a = numpy.einsum('...ij,...j,...jk', u, new_s, vh)
        if bias is not None:
            new_a += bias
        if dtype.kind in 'iu':
            new_a = numpy.rint(new_a)
        return new_a.astype(dtype)

    def setUp(self):
        self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-3, 'eps': 1e-4}
        self.check_double_backward_options = {
            'atol': 1e-3, 'rtol': 1e-3, 'eps': 1e-4}

    def generate_inputs(self):
        a = self.random_matrix(self.shape, self.dtype, scale=(1e-2, 2.0),
                               sym=True)
        return a,

    def forward_expected(self, inputs):
        a, = inputs
        a = 0.5 * (a + a.T)
        y_expect = numpy.linalg.cholesky(a)
        return y_expect.astype(self.dtype),

    def forward(self, inputs, device):
        a, = inputs
        a = 0.5 * (a + a.T)
        y = F.cholesky(a)
        return y,


testing.run_module(__name__, __file__)

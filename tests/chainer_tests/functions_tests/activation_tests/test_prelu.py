import numpy

import chainer
from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(4, 3, 2), (1,), (1, 2, 3, 4, 5, 6)],
    'Wdim': [0, 1, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
@chainer.testing.backend.inject_backend_tests(
    None,
    [
        # NumPy
        {},
        # CuPy
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        # ChainerX
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestPReLU(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-3})

        self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        if self.dtype == numpy.float16:
            self.check_double_backward_options.update(
                {'atol': 5e-3, 'rtol': 5e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        x[(-0.05 < x) & (x < 0.05)] = 0.5
        W = numpy.random.uniform(
            -1, 1, self.shape[1:1 + self.Wdim]).astype(self.dtype)
        return x, W

    def forward_expected(self, inputs):
        x, W = inputs
        y_expect = x.copy()
        masked = numpy.ma.masked_greater_equal(y_expect, 0, copy=False)
        shape = (1,) + W.shape + (1,) * (x.ndim - W.ndim - 1)
        masked *= W.reshape(shape)
        return y_expect,

    def forward(self, inputs, device):
        x, W = inputs
        y = functions.prelu(x, W)
        return y,


testing.run_module(__name__, __file__)

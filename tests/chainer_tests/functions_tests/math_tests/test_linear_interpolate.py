import numpy

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(3, 4), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
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
class TestLinearInterpolate(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({
                'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({
                'atol': 5e-4, 'rtol': 5e-3})
            self.check_double_backward_options.update({
                'atol': 5e-3, 'rtol': 5e-2})

    def generate_inputs(self):
        p = numpy.random.uniform(0, 1, self.shape).astype(self.dtype)
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        y = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return p, x, y

    def forward(self, inputs, device):
        p, x, y = inputs
        ret = functions.linear_interpolate(p, x, y)
        ret = functions.cast(ret, numpy.float64)
        return ret,

    def forward_expected(self, inputs):
        p, x, y = inputs
        expected = p * x + (1 - p) * y
        expected = utils.force_array(expected, dtype=numpy.float64)
        return expected,


testing.run_module(__name__, __file__)

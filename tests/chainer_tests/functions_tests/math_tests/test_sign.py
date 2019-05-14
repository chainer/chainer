import numpy

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestSign(testing.FunctionTestCase):

    skip_backward_test = True
    skip_double_backward_test = True

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-7, 'rtol': 1e-7})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        # Avoid non-differentiable point
        x[(abs(x) < 1e-2)] = 1
        return x,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.sign(x)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        expected = numpy.sign(x)
        expected = utils.force_array(expected)
        return expected,


testing.run_module(__name__, __file__)

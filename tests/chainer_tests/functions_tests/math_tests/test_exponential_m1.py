import numpy

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'shape': [(), (3, 2)],
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
class Expm1FunctionTest(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options.update(
            {'atol': 1e-3, 'rtol': 1e-2})
        self.check_double_backward_options.update(
            {'atol': 1e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.expm1(x),

    def forward_expected(self, inputs):
        x, = inputs
        expected = numpy.expm1(x)
        expected = utils.force_array(expected)
        return expected,


testing.run_module(__name__, __file__)

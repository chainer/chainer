import numpy

from chainer import functions
from chainer import testing
from chainer import utils


def _hard_sigmoid(x):
    return (x * 0.2 + 0.5).clip(0, 1)


@testing.parameterize(*testing.product({
    'shape': [(3, 4), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
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
class TestHardSigmoid(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.hard_sigmoid(x),

    def forward_expected(self, inputs):
        x, = inputs
        y = utils.force_array(_hard_sigmoid(x), self.dtype)
        return y,


testing.run_module(__name__, __file__)

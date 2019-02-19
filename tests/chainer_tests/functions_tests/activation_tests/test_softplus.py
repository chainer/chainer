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
)
class TestSoftplus(testing.FunctionTestCase):

    def setUp(self):
        self.beta = numpy.random.uniform(1, 2, ())
        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {'atol': 5e-2, 'rtol': 5e-1}

    def generate_inputs(self):
        x = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y = numpy.log(1 + numpy.exp(self.beta * x)) / self.beta
        return utils.force_array(y).astype(self.dtype),

    def forward(self, inputs, device):
        x, = inputs
        return functions.softplus(x, beta=self.beta),


testing.run_module(__name__, __file__)

import numpy

from chainer import functions
from chainer import testing
import math


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
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestFmod(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-7, 'rtol': 1e-7})
        self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-3})
        self.check_double_backward_options.update(
            {'atol': 1e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        divisor = numpy.random.uniform(-1.0, 1.0,
                                       self.shape).astype(self.dtype)
        # division with too small divisor is unstable.
        for i in numpy.ndindex(self.shape):
            if math.fabs(divisor[i]) < 0.1:
                divisor[i] += 1.0
        # make enough margin
        for i in numpy.ndindex(self.shape):
            m = math.fabs(x[i] % divisor[i])
            if m < 0.01 or m > (divisor[i] - 0.01):
                x[i] = 0.5
                divisor[i] = 0.3
        return x, divisor

    def forward(self, inputs, device):
        x, divisor = inputs
        y = functions.fmod(x, divisor)
        return y,

    def forward_expected(self, inputs):
        x, divisor = inputs
        expected = numpy.fmod(x, divisor)
        expected = numpy.asarray(expected)
        return expected,


testing.run_module(__name__, __file__)

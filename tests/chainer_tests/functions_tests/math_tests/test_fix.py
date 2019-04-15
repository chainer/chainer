import numpy

from chainer import functions
from chainer import testing


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
class TestFix(testing.FunctionTestCase):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        while True:
            x = numpy.random.uniform(
                -10.0, 10.0, self.shape).astype(self.dtype)
            if (numpy.abs(x - numpy.round(x)) > 1e-2).all():
                return x,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.fix(x)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        expected = numpy.fix(x)
        expected = numpy.asarray(expected)
        return expected,


testing.run_module(__name__, __file__)

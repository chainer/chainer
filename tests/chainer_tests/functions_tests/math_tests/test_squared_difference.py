
import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'in_shape': [(5, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestSquaredDifference(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_double_backward_options \
                .update({'atol': 1e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        x1 = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        x2 = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        return x1, x2

    def forward(self, inputs, device):
        x1, x2 = inputs
        return functions.squared_difference(x1, x2),

    def forward_expected(self, inputs):
        x1, x2 = inputs
        expected = (x1-x2)**2
        expected = numpy.asarray(expected)
        return expected.astype(self.dtype),


testing.run_module(__name__, __file__)

import numpy

from chainer import functions
from chainer import testing
from chainer.utils import force_array


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'function_name': ['cosh', 'sinh'],
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
class TestCoshSinh(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-7, 'rtol': 1e-7})
        if self.dtype == numpy.float16:
            self.check_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-2})
            self.check_double_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-2})
        else:
            self.check_backward_options.update({
                'atol': 1e-4, 'rtol': 1e-3})
            self.check_double_backward_options.update({
                'atol': 1e-4, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        function = getattr(functions, self.function_name)
        return function(x),

    def forward_expected(self, inputs):
        x, = inputs
        function = getattr(numpy, self.function_name)
        expected = function(x)
        expected = force_array(expected)
        return expected,


testing.run_module(__name__, __file__)

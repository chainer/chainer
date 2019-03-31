import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*(testing.product({
    'ch_out': [1, 5],
    'size': [10, 20],
    'dtype': [numpy.float32, numpy.float16],
    'eps': [1e-5, 1e-1],
})))
@testing.backend.inject_backend_tests(
    None,
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestStandardize(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 5e-3, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 5e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        shape = self.ch_out, self.size
        x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.standardize(x, self.eps),

    def forward_expected(self, inputs):
        x, = inputs
        mean = numpy.mean(x, axis=1, keepdims=True)
        var = numpy.mean(numpy.square(x - mean), axis=1, keepdims=True)
        std = numpy.sqrt(var + self.eps, dtype=self.dtype)
        inv_std = 1. / std
        return (x - mean) * inv_std,


testing.run_module(__name__, __file__)

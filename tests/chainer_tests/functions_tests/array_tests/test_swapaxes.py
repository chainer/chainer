import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'in_shape': [(3, 4, 2)],
    'axis1': [0],
    'axis2': [1],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestSwapaxes(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(
            0.5, 1, self.in_shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y_expected = x.swapaxes(self.axis1, self.axis2)
        return y_expected,

    def forward(self, inputs, devices):
        x, = inputs
        y = functions.swapaxes(x, self.axis1, self.axis2)
        return y,


testing.run_module(__name__, __file__)

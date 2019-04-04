import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'in_shape': [(4, 3, 2)],
    'axes': [(-1, 0, 1), None],
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
class TestTranspose(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        return x.transpose(self.axes),

    def forward(self, inputs, device):
        x, = inputs
        y = functions.transpose(x, self.axes)
        return y,


testing.run_module(__name__, __file__)

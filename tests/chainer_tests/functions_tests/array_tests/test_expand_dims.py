import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product_dict(
    [
        {'in_shape': (3, 2), 'out_shape': (1, 3, 2), 'axis': 0},
        {'in_shape': (3, 2), 'out_shape': (3, 1, 2), 'axis': 1},
        {'in_shape': (3, 2), 'out_shape': (3, 2, 1), 'axis': 2},
        {'in_shape': (3, 2), 'out_shape': (3, 2, 1), 'axis': -1},
        {'in_shape': (3, 2), 'out_shape': (3, 1, 2), 'axis': -2},
        {'in_shape': (3, 2), 'out_shape': (1, 3, 2), 'axis': -3},
        {'in_shape': (3, 2), 'out_shape': (1, 3, 2), 'axis': 0},
        {'in_shape': (3, 2), 'out_shape': (1, 3, 2), 'axis': 0},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
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
class TestExpandDims(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y_expect = numpy.expand_dims(cuda.to_cpu(x), self.axis)
        return y_expect,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.expand_dims(x, self.axis)
        return y,


testing.run_module(__name__, __file__)

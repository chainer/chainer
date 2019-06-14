import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (5, 4), 'y_shape': (10, 4), 'axis': 0},
        {'shape': (5, 4), 'y_shape': (5, 8), 'axis': 1},
        {'shape': (5, 4), 'y_shape': (5, 8), 'axis': -1},
        {'shape': (5, 4), 'y_shape': (10, 4), 'axis': -2},
        {'shape': (5, 4, 3, 2), 'y_shape': (10, 4, 3, 2), 'axis': 0},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 8, 3, 2), 'axis': 1},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 6, 2), 'axis': 2},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 3, 4), 'axis': 3},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 3, 4), 'axis': -1},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 4, 6, 2), 'axis': -2},
        {'shape': (5, 4, 3, 2), 'y_shape': (5, 8, 3, 2), 'axis': -3},
        {'shape': (5, 4, 3, 2), 'y_shape': (10, 4, 3, 2), 'axis': -4},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestCReLU(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.crelu(x, axis=self.axis),

    def forward_expected(self, inputs):
        x, = inputs
        expected_former = numpy.maximum(x, 0)
        expected_latter = numpy.maximum(-x, 0)
        expected = numpy.concatenate(
            (expected_former, expected_latter), axis=self.axis)
        return expected,


testing.run_module(__name__, __file__)

import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (3, 4), 'axis': 0, 'y_shape': (2, 3, 4)},
        {'shape': (3, 4), 'axis': 1, 'y_shape': (3, 2, 4)},
        {'shape': (3, 4), 'axis': 2, 'y_shape': (3, 4, 2)},
        {'shape': (3, 4), 'axis': -1, 'y_shape': (3, 4, 2)},
        {'shape': (3, 4), 'axis': -2, 'y_shape': (3, 2, 4)},
        {'shape': (3, 4), 'axis': -3, 'y_shape': (2, 3, 4)},
        {'shape': (), 'axis': 0, 'y_shape': (2,)},
        {'shape': (), 'axis': -1, 'y_shape': (2,)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
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
        'use_cudnn': ['always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestStack(testing.FunctionTestCase):

    def generate_inputs(self):
        xs = tuple([
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype),
        ])
        return xs

    def forward_expected(self, inputs):
        xs = inputs
        if hasattr(numpy, 'stack'):
            # run test only with numpy>=1.10
            expect = numpy.stack(xs, axis=self.axis)
        else:
            raise Exception('test only with numpy>=1.10')
        return expect,

    def forward(self, inputs, device):
        xs = inputs
        y = functions.stack(xs, axis=self.axis)
        return y,


testing.run_module(__name__, __file__)

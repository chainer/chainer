import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 4, 6), 'args': (1, 2, 0)},
        {'shape': (2, 4, 6), 'args': (-1, 2, 0)},
        {'shape': (2, 4, 6), 'args': (0, -1, -2)},
        {'shape': (2, 4, 6), 'args': (0, -1, 1)},
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
    testing.product({
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + [{'use_cuda': True}]
)
class TestDiagonal(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.diagonal(x, *self.args),

    def forward_expected(self, inputs):
        x, = inputs
        return x.diagonal(*self.args),


testing.run_module(__name__, __file__)

import unittest

import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'in_shape': [(), 2, (2, 3)],
    'reps': [(), 0, 2, (0, 0), (1, 2), (2, 2), (2, 0)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
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
class TestTile(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options = {}
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = ({'atol': 5e-4, 'rtol': 5e-3})
            self.check_backward_options = ({'atol': 2 ** -4, 'rtol': 2 ** -4})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y_expected = numpy.tile(x, self.reps)
        return y_expected,

    def forward(self, inputs, devices):
        x, = inputs
        y = functions.tile(x, self.reps)
        return y,


@testing.parameterize(*testing.product({
    'reps': [-1, (-1, -1)],
}))
class TestTileValueError(unittest.TestCase):

    def test_value_error(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(ValueError):
            functions.tile(x, self.reps)


class TestTileTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)

    def test_reps_not_int(self):
        with self.assertRaises(TypeError):
            functions.tile(self.x, 'a')

    def test_x_not_ndarray_or_variable(self):
        with self.assertRaises(TypeError):
            functions.tile((self.x, self.x), 2)


testing.run_module(__name__, __file__)

import unittest

import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'axis': [None, 0, 1, 2, -1, (0, 1), (1, 0), (0, -1), (-2, 0)],
    'keepdims': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestSum(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_double_backward_options \
                .update({'atol': 1e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.sum(x, axis=self.axis, keepdims=self.keepdims),

    def forward_expected(self, inputs):
        x, = inputs
        expected = x.sum(axis=self.axis, keepdims=self.keepdims)
        expected = numpy.asarray(expected)
        return expected,


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSumError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.sum(self.x, axis=[0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.sum(self.x, axis=(1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.sum(self.x, axis=(0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.x.sum(axis=(1, -2))


testing.run_module(__name__, __file__)

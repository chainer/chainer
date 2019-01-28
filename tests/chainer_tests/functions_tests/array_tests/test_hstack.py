import unittest

import numpy
import six

from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3, 4), 'y_shape': (2, 6, 4), 'xs_length': 2},
        {'shape': (3, 4), 'y_shape': (3, 8), 'xs_length': 2},
        {'shape': (3), 'y_shape': (6,), 'xs_length': 2},
        {'shape': (), 'y_shape': (2,), 'xs_length': 2},
        {'shape': (2, 3, 4), 'y_shape': (2, 3, 4), 'xs_length': 1},
        {'shape': (3, 4), 'y_shape': (3, 4), 'xs_length': 1},
        {'shape': (3), 'y_shape': (3,), 'xs_length': 1},
        {'shape': (), 'y_shape': (1,), 'xs_length': 1},
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
        {'use_ideep': 'always'},
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
class TestHstack(testing.FunctionTestCase):

    def generate_inputs(self):
        return tuple([
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            for i in six.moves.range(self.xs_length)])

    def forward(self, inputs, device):
        y = functions.hstack(inputs)
        return y,

    def forward_expected(self, inputs):
        y = numpy.hstack(inputs)
        return y,


@testing.parameterize(
    {'a_shape': (2, 4, 5), 'b_shape': (3, 4, 5), 'valid': False},
    {'a_shape': (3, 4, 6), 'b_shape': (3, 4, 5), 'valid': False},
    {'a_shape': (3, 6, 5), 'b_shape': (3, 4, 5), 'valid': True},
    {'a_shape': (3, 4), 'b_shape': (4, 4), 'valid': False},
    {'a_shape': (3, 4), 'b_shape': (3, 3), 'valid': True},
    {'a_shape': (3,), 'b_shape': (4,), 'valid': True},
    {'a_shape': (3), 'b_shape': (3, 3), 'valid': False},
)
class TestHstackTypeCheck(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.a_shape).astype(numpy.float32),
            numpy.random.uniform(-1, 1, self.b_shape).astype(numpy.float32),
        ]

    def check_value_check(self):
        if self.valid:
            # Check if it throws nothing
            functions.hstack(self.xs)
        else:
            with self.assertRaises(type_check.InvalidType):
                functions.hstack(self.xs)

    def test_value_check_cpu(self):
        self.check_value_check()

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check()


testing.run_module(__name__, __file__)

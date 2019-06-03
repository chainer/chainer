import unittest

import numpy
import pytest
import six

from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3, 4), 'y_shape': (4, 3, 4), 'xs_length': 2},
        {'shape': (3, 4), 'y_shape': (6, 4), 'xs_length': 2},
        {'shape': (3), 'y_shape': (2, 3), 'xs_length': 2},
        {'shape': (), 'y_shape': (2, 1), 'xs_length': 2},
        {'shape': (2, 3, 4), 'y_shape': (2, 3, 4), 'xs_length': 1},
        {'shape': (3, 4), 'y_shape': (3, 4), 'xs_length': 1},
        {'shape': (3), 'y_shape': (1, 3), 'xs_length': 1},
        {'shape': (), 'y_shape': (1, 1), 'xs_length': 1},
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
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestVstack(testing.FunctionTestCase):

    def setup(self):
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def generate_inputs(self):
        xs = tuple(
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            for _ in six.moves.range(self.xs_length)
        )
        return xs

    def forward_expected(self, inputs):
        x = list(inputs)
        y_expect = numpy.vstack(x)
        return y_expect,

    def forward(self, inputs, device):
        x = list(inputs)
        y = functions.vstack(x)
        return y,


@testing.parameterize(
    {'a_shape': (2, 4, 5), 'b_shape': (3, 4, 5), 'valid': True},
    {'a_shape': (3, 4, 6), 'b_shape': (3, 4, 5), 'valid': False},
    {'a_shape': (3, 6, 5), 'b_shape': (3, 4, 5), 'valid': False},
    {'a_shape': (3, 4), 'b_shape': (4, 4), 'valid': True},
    {'a_shape': (3, 4), 'b_shape': (3, 3), 'valid': False},
    {'a_shape': (3,), 'b_shape': (4,), 'valid': False},
    {'a_shape': (3), 'b_shape': (3, 3), 'valid': False},
    {'a_shape': (), 'b_shape': (1), 'valid': False},
)
class TestVstackTypeCheck(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, self.a_shape).astype(numpy.float32),
            numpy.random.uniform(-1, 1, self.b_shape).astype(numpy.float32),
        ]

    def check_value_check(self):
        if self.valid:
            # Check if it throws nothing
            functions.vstack(self.xs)
        else:
            with pytest.raises(type_check.InvalidType):
                functions.vstack(self.xs)

    def test_value_check_cpu(self):
        self.check_value_check()

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check()


testing.run_module(__name__, __file__)

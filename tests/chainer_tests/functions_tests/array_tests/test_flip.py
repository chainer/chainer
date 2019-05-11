import unittest

import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (1,), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 2},
        {'shape': (2, 3, 4), 'axis': -3},
        {'shape': (2, 3, 4), 'axis': -2},
        {'shape': (2, 3, 4), 'axis': -1},
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
class TestFlip(testing.FunctionTestCase):

    def setUp(self):
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        flip_func = getattr(numpy, 'flip', functions.array.flip._flip)
        y_expected = flip_func(x, self.axis)
        return y_expected,

    def forward(self, inputs, devices):
        x, = inputs
        y = functions.flip(x, self.axis)
        return y,


@testing.parameterize(
    {'axis': 3},
    {'axis': -4},
)
class TestFlipInvalidTypeAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(type_check.InvalidType):
            functions.flip(x, self.axis)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


class TestFlipInvalidTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def test_invalid_axis(self):
        with self.assertRaises(TypeError):
            functions.flip(self.x, 'a')


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (1,), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 2},
        {'shape': (2, 3, 4), 'axis': -3},
        {'shape': (2, 3, 4), 'axis': -2},
        {'shape': (2, 3, 4), 'axis': -1},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
@testing.with_requires('numpy>=1.12.0')
class TestFlipFunction(unittest.TestCase):

    def test_equal_to_numpy_flip(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        numpy.testing.assert_array_equal(
            functions.array.flip._flip(x, self.axis),
            numpy.flip(x, self.axis))


testing.run_module(__name__, __file__)

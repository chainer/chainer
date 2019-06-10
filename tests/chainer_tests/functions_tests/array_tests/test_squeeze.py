import unittest

import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


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
@testing.parameterize(*testing.product_dict(
    [
        {'axis': None, 'out_shape': (3,)},
        {'axis': 1, 'out_shape': (1, 3, 1)},
        {'axis': -3, 'out_shape': (1, 3, 1)},
        {'axis': (0, 1, 3), 'out_shape': (3,)},
        {'axis': (3, 1, 0), 'out_shape': (3,)},
        {'axis': (-4, -3, -1), 'out_shape': (3,)},
        {'axis': (-1, -3, -4), 'out_shape': (3,)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestSqueeze(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-4, 'rtol': 5e-3})
            self.check_backward_options.update({
                'atol': 2 ** -4, 'rtol': 2 ** -4})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, (1, 1, 3, 1)).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y = numpy.squeeze(x, axis=self.axis)
        return y,

    def forward(self, inputs, device):
        x, = inputs
        return functions.squeeze(x, axis=self.axis),


@testing.parameterize(*testing.product(
    {'axis': [1, (1,)]},
))
class TestSqueezeValueError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 3, 1)).astype('f')

    def check_invalid_type(self, x_data):
        with self.assertRaises(ValueError):
            functions.squeeze(x_data, axis=self.axis)

    def test_invalid_type_cpu(self):
        self.check_invalid_type(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_invalid_type(cuda.to_gpu(self.x))


@testing.parameterize(*testing.product(
    {'axis': [3, -4, (3,), (-4,)]},
))
class TestSqueezeInvalidType(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 3, 1)).astype('f')

    def check_invalid_type(self, x_data):
        with self.assertRaises(type_check.InvalidType):
            functions.squeeze(x_data, axis=self.axis)

    def test_invalid_type_cpu(self):
        self.check_invalid_type(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_invalid_type(cuda.to_gpu(self.x))


class TestSqueezeTypeError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 3, 1)).astype('f')

    def test_invalid_axis(self):
        with self.assertRaises(TypeError):
            functions.squeeze(self.x, axis='a')


testing.run_module(__name__, __file__)

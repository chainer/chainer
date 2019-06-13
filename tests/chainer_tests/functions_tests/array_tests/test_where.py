import unittest

import numpy
import pytest

import chainer
from chainer import functions
from chainer import testing
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'shape': [
        # c, x, y, output
        ((3, 2, 4),) * 4,
        ((4,), (3, 1, 1), (2, 1), (3, 2, 4)),
    ],
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
class TestWhere(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-3,
            })

    def generate_inputs(self):
        c_shape, x_shape, y_shape, out_shape = self.shape
        c = numpy.random.uniform(-1, 1, c_shape) > 0
        x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        y = numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)
        return c, x, y

    def forward_expected(self, inputs):
        c, x, y = inputs
        z_expected = numpy.where(c, x, y)
        return z_expected,

    def forward(self, inputs, devices):
        c, x, y = inputs
        z = functions.where(c, x, y)
        return z,


class TestWhereTypeCheck(unittest.TestCase):

    def check_forward_raises(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        with pytest.raises(type_check.InvalidType):
            functions.where(c, x, y)

    def test_cond_int(self):
        c_data = numpy.zeros(3, numpy.int32)
        x_data = numpy.zeros(3, numpy.float32)
        y_data = numpy.zeros(3, numpy.float32)
        self.check_forward_raises(c_data, x_data, y_data)

    def test_xy_precision(self):
        c_data = numpy.zeros(3, numpy.bool_)
        x_data = numpy.zeros(3, numpy.float32)
        y_data = numpy.zeros(3, numpy.float64)
        self.check_forward_raises(c_data, x_data, y_data)

    def test_shape(self):
        c_data = numpy.zeros(3, numpy.bool_)
        x_data = numpy.zeros(1, numpy.float32)
        y_data = numpy.zeros(2, numpy.float32)
        self.check_forward_raises(c_data, x_data, y_data)


testing.run_module(__name__, __file__)

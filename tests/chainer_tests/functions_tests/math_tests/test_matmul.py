import unittest

import numpy

import chainer
from chainer.backend import CpuDevice
import chainer.functions as F
from chainer import testing
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [
        # matmul
        {'x1_shape': (2, 5), 'x2_shape': (5, 10),
         'transa': False, 'transb': False},
        {'x1_shape': (5, 2), 'x2_shape': (5, 10),
         'transa': True, 'transb': False},
        {'x1_shape': (2, 5), 'x2_shape': (10, 5),
         'transa': False, 'transb': True},
        {'x1_shape': (5, 2), 'x2_shape': (10, 5),
         'transa': True, 'transb': True},

        # vector
        {'x1_shape': (5,), 'x2_shape': (5,),
         'transa': True, 'transb': False},
        {'x1_shape': (5,), 'x2_shape': (5,),
         'transa': False, 'transb': True},

        # matrix-vector
        {'x1_shape': (5,), 'x2_shape': (5, 2),
         'transa': False, 'transb': False},
        {'x1_shape': (5,), 'x2_shape': (5, 2),
         'transa': True, 'transb': False},
        {'x1_shape': (5,), 'x2_shape': (2, 5),
         'transa': False, 'transb': True},
        {'x1_shape': (2, 5), 'x2_shape': (5,),
         'transa': False, 'transb': False},
        {'x1_shape': (5, 2), 'x2_shape': (5,),
         'transa': True, 'transb': False},
        {'x1_shape': (2, 5), 'x2_shape': (5,),
         'transa': False, 'transb': True},

        # batched matmul
        {'x1_shape': (6, 2, 5), 'x2_shape': (6, 5, 10),
         'transa': False, 'transb': False},
        {'x1_shape': (6, 5, 2), 'x2_shape': (6, 5, 10),
         'transa': True, 'transb': False},
        {'x1_shape': (6, 2, 5), 'x2_shape': (6, 10, 5),
         'transa': False, 'transb': True},
        {'x1_shape': (6, 5, 2), 'x2_shape': (6, 10, 5),
         'transa': True, 'transb': True},
        {'x1_shape': (2, 3, 4), 'x2_shape': (4,),
         'transa': False, 'transb': False},
        {'x1_shape': (4,), 'x2_shape': (2, 4, 3),
         'transa': False, 'transb': False},

        # batchsize = 1
        {'x1_shape': (1, 2, 5), 'x2_shape': (1, 5, 10),
         'transa': False, 'transb': False},

        # 4dim batched matmul
        {'x1_shape': (2, 3, 4, 5), 'x2_shape': (2, 3, 5, 6),
         'transa': False, 'transb': False},
    ],
    [
        {'x1_dtype': numpy.float16},
        {'x1_dtype': numpy.float32},
        {'x1_dtype': numpy.float64},
    ],
    [
        {'x2_dtype': numpy.float16},
        {'x2_dtype': numpy.float32},
        {'x2_dtype': numpy.float64},
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
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']
    })
)
class TestMatMul(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-5}
        if self.x1_dtype == numpy.float16 or self.x2_dtype == numpy.float16:
            self.check_forward_options = {'atol': 2e-3, 'rtol': 2e-3}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        x1 = x1 = numpy.random.uniform(.5, 1, self.x1_shape)
        x1 = x1.astype(self.x1_dtype)
        x2 = numpy.random.uniform(.5, 1, self.x2_shape)
        x2 = x2.astype(self.x2_dtype)
        return x1, x2

    def forward_expected(self, inputs):
        x1, x2 = inputs
        if self.transa and x1.ndim >= 2:
            x1 = x1.swapaxes(-1, -2)
        if self.transb and x2.ndim >= 2:
            x2 = x2.swapaxes(-1, -2)
        if x1.ndim <= 2 or x2.ndim <= 2:
            y = numpy.dot(x1, x2)
            device = CpuDevice()
            y = device.send(y)
        else:
            y = numpy.einsum('...ij,...jk->...ik', x1, x2)
        return y,

    def forward(self, inputs, device):
        x1, x2 = inputs
        y = F.matmul(x1, x2, transa=self.transa, transb=self.transb)
        return y,


@testing.parameterize(*testing.product_dict(
    [
        # batched matmul 2d x 2d
        {'x1_shape': (2, 3), 'x2_shape': (2, 3),
         'transa': True, 'transb': False},
        {'x1_shape': (2, 3), 'x2_shape': (2, 3),
         'transa': False, 'transb': True},

        # batched matmul 3d x 3d
        {'x1_shape': (3, 2, 5), 'x2_shape': (3, 5, 4),
         'transa': False, 'transb': False},
        {'x1_shape': (3, 5, 2), 'x2_shape': (3, 5, 4),
         'transa': True, 'transb': False},
        {'x1_shape': (3, 2, 5), 'x2_shape': (3, 4, 5),
         'transa': False, 'transb': True},
        {'x1_shape': (3, 5, 2), 'x2_shape': (3, 4, 5),
         'transa': True, 'transb': True},

        # batched matmul 2d x 3d
        {'x1_shape': (3, 5), 'x2_shape': (3, 1, 4),
         'transa': False, 'transb': False},
        {'x1_shape': (3, 5), 'x2_shape': (3, 5, 4),
         'transa': True, 'transb': False},
        {'x1_shape': (3, 5), 'x2_shape': (3, 4, 1),
         'transa': False, 'transb': True},
        {'x1_shape': (3, 5), 'x2_shape': (3, 4, 5),
         'transa': True, 'transb': True},

        # batched matmul 3d x 2d
        {'x1_shape': (3, 2, 5), 'x2_shape': (3, 5),
         'transa': False, 'transb': False},
        {'x1_shape': (3, 5, 2), 'x2_shape': (3, 5),
         'transa': True, 'transb': False},
        {'x1_shape': (3, 2, 1), 'x2_shape': (3, 5),
         'transa': False, 'transb': True},
        {'x1_shape': (3, 1, 2), 'x2_shape': (3, 5),
         'transa': True, 'transb': True},

        # batchsize = 1
        {'x1_shape': (1, 2, 5), 'x2_shape': (1, 5, 4),
         'transa': False, 'transb': False},
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
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']
    })
)
class TestBatchMatMul(testing.FunctionTestCase):
    x1_dtype = numpy.float32
    x2_dtype = numpy.float32

    def generate_inputs(self):
        x1 = numpy.random.uniform(.5, 1, self.x1_shape)
        x1 = x1.astype(self.x1_dtype)
        x2 = numpy.random.uniform(.5, 1, self.x2_shape)
        x2 = x2.astype(self.x2_dtype)
        return x1, x2

    def forward_expected(self, inputs):
        x1, x2 = inputs
        x1 = x1.reshape(x1.shape[:2] + (-1,))
        if self.transa:
            x1 = x1.swapaxes(-1, -2)

        x2 = x2.reshape(x2.shape[:2] + (-1,))
        if self.transb:
            x2 = x2.swapaxes(-1, -2)

        y_expect = numpy.einsum('...ij,...jk->...ik', x1, x2)
        return y_expect,

    def forward(self, inputs, device):
        x1, x2 = inputs
        with testing.assert_warns(DeprecationWarning):
            y = F.batch_matmul(
                x1, x2, transa=self.transa, transb=self.transb)
        return y,


class TestMatMulInvalid(unittest.TestCase):

    def test_invalid_shape(self):
        x_data = numpy.zeros((2, 3, 4), dtype=numpy.float32)
        y_data = numpy.zeros((3, 4, 3), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            F.matmul(x, y)

    def test_invalid_ndim(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.float32)
        y_data = numpy.zeros((3, 5), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            F.matmul(x, y)


testing.run_module(__name__, __file__)

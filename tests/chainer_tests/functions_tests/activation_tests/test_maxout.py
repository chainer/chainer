import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


def _maxout(x, pool_size, axis):
    shape = (x.shape[:axis] + (x.shape[axis] // pool_size, pool_size) +
             x.shape[axis + 1:])
    x = x.reshape(shape)
    return x.max(axis=axis + 1)


@testing.parameterize(*testing.product_dict(
    [
        {'x_shape': (7, 12), 'pool_size': 2, 'axis': 1, 'y_shape': (7, 6)},
        {'x_shape': (7, 12), 'pool_size': 12, 'axis': 1, 'y_shape': (7, 1)},
        {'x_shape': (7, 3, 4), 'pool_size': 7, 'axis': 0,
         'y_shape': (1, 3, 4)},
        {'x_shape': (7, 3, 4), 'pool_size': 3, 'axis': 1,
         'y_shape': (7, 1, 4)},
        {'x_shape': (7, 3, 4), 'pool_size': 4, 'axis': 2,
         'y_shape': (7, 3, 1)},
        {'x_shape': (7, 2, 3, 4), 'pool_size': 2, 'axis': 3,
         'y_shape': (7, 2, 3, 2)},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
@testing.fix_random()
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
class TestNonparameterizedMaxout(testing.FunctionTestCase):
    dodge_nondifferentiable = True

    def generate_inputs(self):
        x_size = numpy.prod(self.x_shape)
        x = numpy.random.permutation(numpy.arange(x_size))\
            .reshape(self.x_shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.maxout(x, self.pool_size, self.axis),

    def forward_expected(self, inputs):
        x, = inputs
        return _maxout(x, self.pool_size, self.axis),


@testing.parameterize(
    {'x_shape': (2, 3, 4), 'pool_size': 5, 'error': type_check.InvalidType},
    {'x_shape': (2, 3, 4), 'pool_size': -1, 'error': ValueError}
)
class InvalidArgument(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(
            numpy.random.uniform(-1, 1, self.x_shape).astype(numpy.float32))

    def test_invalid_shape_cpu(self):
        with self.assertRaises(self.error):
            functions.maxout(self.x, self.pool_size)

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.x.to_gpu()
        with self.assertRaises(self.error):
            functions.maxout(self.x, self.pool_size)


testing.run_module(__name__, __file__)

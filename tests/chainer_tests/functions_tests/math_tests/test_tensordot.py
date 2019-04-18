import unittest

import numpy

import chainer
import chainer.functions as F
from chainer import testing


@testing.parameterize(*testing.product_dict(
    [
        {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': 2, 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': ([1, 2], [0, 1]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (4, 2, 3), 'b_shape': (3, 5, 2), 'axes': ([2, 1], [0, 2]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (2, 4, 3), 'b_shape': (5, 3, 2), 'axes': ([2, 0], [1, 2]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (2, 3, 4), 'b_shape': (5, 2, 3), 'axes': ([1, 0], [2, 1]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (3, 2, 4), 'b_shape': (2, 5, 3), 'axes': ([0, 1], [2, 0]), 'gc_shape': (4, 5)},  # NOQA
        {'a_shape': (3, 4, 2), 'b_shape': (2, 3, 5), 'axes': ([0, 2], [1, 0]), 'gc_shape': (4, 5)},  # NOQA

        {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': 1, 'gc_shape': (3, 4, 5, 6)},  # NOQA
        {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': ([2, 0]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
        {'a_shape': (3, 2, 4), 'b_shape': (5, 2, 6), 'axes': ([1, 1]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
        {'a_shape': (2, 3, 4), 'b_shape': (5, 6, 2), 'axes': ([0, 2]), 'gc_shape': (3, 4, 5, 6)},  # NOQA

        {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': 2, 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': ([2, 3], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (4, 5, 2, 3), 'b_shape': (3, 6, 2), 'axes': ([3, 2], [0, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (4, 2, 5, 3), 'b_shape': (6, 3, 2), 'axes': ([3, 1], [1, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (2, 4, 5, 3), 'b_shape': (6, 2, 3), 'axes': ([3, 0], [2, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (2, 4, 3, 5), 'b_shape': (2, 6, 3), 'axes': ([2, 0], [2, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (2, 3, 4, 5), 'b_shape': (2, 3, 6), 'axes': ([1, 0], [1, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (3, 2, 4, 5), 'b_shape': (3, 2, 6), 'axes': ([0, 1], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
        {'a_shape': (3, 2, 5, 4), 'b_shape': (3, 6, 2), 'axes': ([0, 1], [0, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
        {'a_shape': (3, 5, 2, 4), 'b_shape': (6, 3, 2), 'axes': ([0, 2], [1, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
        {'a_shape': (5, 3, 2, 4), 'b_shape': (6, 2, 3), 'axes': ([1, 2], [2, 1]), 'gc_shape': (5, 4, 6)},  # NOQA

        {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': 3, 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': ([1, 2, 3], [0, 1, 2]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (5, 4, 2, 3), 'b_shape': (4, 3, 6, 2), 'axes': ([1, 3, 2], [0, 1, 3]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (5, 2, 4, 3), 'b_shape': (4, 6, 3, 2), 'axes': ([2, 3, 1], [0, 2, 3]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (2, 5, 4, 3), 'b_shape': (4, 6, 2, 3), 'axes': ([2, 3, 0], [0, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (2, 5, 3, 4), 'b_shape': (6, 4, 2, 3), 'axes': ([3, 2, 0], [1, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (2, 3, 5, 4), 'b_shape': (6, 2, 4, 3), 'axes': ([3, 1, 0], [2, 3, 1]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (3, 2, 5, 4), 'b_shape': (6, 2, 3, 4), 'axes': ([3, 0, 1], [3, 2, 1]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (3, 2, 4, 5), 'b_shape': (2, 6, 3, 4), 'axes': ([2, 0, 1], [3, 2, 0]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (3, 4, 2, 5), 'b_shape': (2, 3, 6, 4), 'axes': ([1, 0, 2], [3, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (4, 3, 2, 5), 'b_shape': (2, 3, 4, 6), 'axes': ([0, 1, 2], [2, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (4, 3, 5, 2), 'b_shape': (3, 2, 4, 6), 'axes': ([0, 1, 3], [2, 0, 1]), 'gc_shape': (5, 6)},  # NOQA
        {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 4, 2, 6), 'axes': ([0, 2, 3], [1, 0, 2]), 'gc_shape': (5, 6)},  # NOQA

        {'a_shape': (3, 2), 'b_shape': (2, 4), 'axes': 1, 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (3, 2), 'b_shape': (2, 4), 'axes': (1, 0), 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (3, 2), 'b_shape': (4, 2), 'axes': (1, 1), 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (4, 2), 'axes': (0, 1), 'gc_shape': (3, 4)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (2, 4), 'axes': (0, 0), 'gc_shape': (3, 4)},  # NOQA

        {'a_shape': (), 'b_shape': (), 'axes': 0, 'gc_shape': ()},  # NOQA
        {'a_shape': (2), 'b_shape': (3), 'axes': 0, 'gc_shape': (2, 3)},  # NOQA
        {'a_shape': (), 'b_shape': (2, 3), 'axes': 0, 'gc_shape': (2, 3)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (), 'axes': 0, 'gc_shape': (2, 3)},  # NOQA
        {'a_shape': (2, 3), 'b_shape': (4), 'axes': 0, 'gc_shape': (2, 3, 4)},  # NOQA
        {'a_shape': (2), 'b_shape': (3, 4), 'axes': 0, 'gc_shape': (2, 3, 4)},  # NOQA
    ],
    [
        {'a_dtype': numpy.float16},
        {'a_dtype': numpy.float32},
        {'a_dtype': numpy.float64},
    ],
    [
        {'b_dtype': numpy.float16},
        {'b_dtype': numpy.float32},
        {'b_dtype': numpy.float64},
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
class TestTensorDot(testing.FunctionTestCase):

    def setUp(self):
        if self.a_dtype == numpy.float16 or self.b_dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 1e-2})

    def generate_inputs(self):
        a = self._setup_tensor(.5, 1, self.a_shape, self.a_dtype)
        b = self._setup_tensor(.5, 1, self.b_shape, self.b_dtype)
        return a, b

    def forward_expected(self, inputs):
        a, b = inputs
        y_expect = numpy.tensordot(a, b, self.axes)
        return y_expect,

    def forward(self, inputs, device):
        a, b = inputs
        y = F.tensordot(a, b, axes=self.axes)
        return y,

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)


class TestTensorDotInvalid(unittest.TestCase):

    def test_invalid_shape(self):
        a_data = numpy.zeros((4, 3, 2), dtype=numpy.float32)
        b_data = numpy.zeros((2, 3, 5), dtype=numpy.float32)
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        with self.assertRaises(ValueError):
            F.tensordot(a, b)
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((1, 2), (0, 1)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((0), (0)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((2), (2)))

    def test_invalid_axes(self):
        a_data = numpy.zeros((4, 3, 2), dtype=numpy.float32)
        b_data = numpy.zeros((3, 2, 5), dtype=numpy.float32)
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((1, 2), (0)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((2), (0, 1)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((0, 1, 2, 3), (0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=(()))
        with self.assertRaises(ValueError):
            F.tensordot(a, b, axes=((), (), ()))
        with self.assertRaises(TypeError):
            F.tensordot(a, b, axes=1.0)


testing.run_module(__name__, __file__)

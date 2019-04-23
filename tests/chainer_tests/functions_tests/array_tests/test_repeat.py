import unittest

import numpy

from chainer import functions
from chainer import testing


def _repeat(arr, repeats, axis=None):
    # Workaround NumPy 1.9 issue.
    if isinstance(repeats, tuple) and len(repeats) == 1:
        repeats = repeats[0]
    return numpy.repeat(arr, repeats, axis)


@testing.parameterize(*testing.product({
    # repeats is any of (int, bool or tuple) and
    # axis is any of (int or None).
    'params': (
        # Repeats 1-D array
        testing.product({
            'shape': [(2,)],
            'repeats': [0, 1, 2, True, (0,), (1,), (2,), (True,)],
            'axis': [None, 0],
        }) +
        # Repeats 2-D array (with axis=None)
        testing.product({
            'shape': [(3, 2)],
            'repeats': [4, (4,), (4,) * 6, (True,) * 6],
            'axis': [None],
        }) +
        # Repeats 2-D array (with axis=0)
        testing.product({
            'shape': [(3, 2)],
            'repeats': [5, (5,), (5,) * 3],
            'axis': [0],
        }) +
        # Repeats 2-D array (with axis=1)
        testing.product({
            'shape': [(3, 2)],
            'repeats': [5, (5,), (5,) * 2],
            'axis': [1],
        }) +
        # Repeats 3-D array (with axis=-2)
        testing.product({
            'shape': [(3, 2, 4)],
            'repeats': [5, (5,), (5,) * 2],
            'axis': [-2],
        })
    ),
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
class TestRepeat(testing.FunctionTestCase):

    def setUp(self):
        self.in_shape = self.params['shape']
        self.repeats = self.params['repeats']
        self.axis = self.params['axis']

        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-4, 'rtol': 5e-3})
            self.check_backward_options.update({
                'atol': 2 ** -4, 'rtol': 2 ** -4})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y_expected = _repeat(x, self.repeats, self.axis)
        return y_expected,

    def forward(self, inputs, devices):
        x, = inputs
        y = functions.repeat(x, self.repeats, self.axis)
        return y,


@testing.parameterize(*testing.product({
    'repeats': [-1, (-1, -1)],
    'axis': [-1],
}))
class TestRepeatValueError(unittest.TestCase):

    def test_value_error(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(ValueError):
            functions.repeat(x, self.repeats, self.axis)


class TestRepeatTypeError(unittest.TestCase):

    def test_type_error_repeats_str(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(TypeError):
            functions.repeat(x, 'a')

    def test_type_error_axis_str(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(TypeError):
            functions.repeat(x, 1, 'a')

    def test_type_error_axis_bool(self):
        x = numpy.random.uniform(-1, 1, (2,)).astype(numpy.float32)
        with self.assertRaises(TypeError):
            functions.repeat(x, 1, True)


testing.run_module(__name__, __file__)

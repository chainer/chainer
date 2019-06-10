import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'function_name': ['max', 'min'],
    'shape': [(3, 2, 4)],
    'dtype': [numpy.float32],
    'axis': [
        None,
        0, 1, 2,  # axis
        -1,  # negative_axis
        (0, 1),  # multi_axis
        (1, 0),  # multi_axis_invert
        (0, -1),  # negative_multi_axis
        (-2, 0),  # negative_multi_axis_invert
    ],
    'keepdims': [True, False],
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
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestMinMax(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options.update({
            'eps': 1e-5, 'atol': 1e-3, 'rtol': 1e-2})
        self.check_double_backward_options.update({
            'eps': 1e-5, 'atol': 1e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        eps = 1e-5

        # Sample x with single maximum/minimum value
        while True:
            x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
            if self.function_name == 'max':
                y = x.max(axis=self.axis, keepdims=True)
                if not numpy.all((x > y - 2 * eps).sum(axis=self.axis) == 1):
                    continue
            elif self.function_name == 'min':
                y = x.min(axis=self.axis, keepdims=True)
                if not numpy.all((x < y + 2 * eps).sum(axis=self.axis) == 1):
                    continue
            return x,

    def forward(self, inputs, device):
        x, = inputs
        function = getattr(functions, self.function_name)
        y = function(x, axis=self.axis, keepdims=self.keepdims)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        function = getattr(numpy, 'a' + self.function_name)
        expected = function(x, axis=self.axis, keepdims=self.keepdims)
        expected = utils.force_array(expected)
        return expected,


@testing.parameterize(*testing.product({
    'function_name': ['max', 'min'],
}))
class TestMinMaxInvalid(unittest.TestCase):

    def setUp(self):
        self.function = getattr(functions, self.function_name)
        self.x = numpy.array([1], dtype=numpy.float32)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            self.function(self.x, [0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            self.function(self.x, (1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            self.function(self.x, (0, 0))

    def test_pos_neg_duplicate_axis(self):
        x_data = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        x = chainer.Variable(x_data)
        with self.assertRaises(ValueError):
            self.function(x, axis=(1, -2))


@testing.parameterize(*testing.product({
    'function_name': ['argmax', 'argmin'],
    'axis': [None, 0, 1, 2, -1, -2, -3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(3, 2, 4)],
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
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestArgMinMax(testing.FunctionTestCase):

    skip_backward_test = True
    skip_double_backward_test = True

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        function = getattr(functions, self.function_name)
        y = function(x, axis=self.axis)
        y = functions.cast(y, numpy.int64)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        function = getattr(numpy, self.function_name)
        expected = function(x, axis=self.axis)
        expected = utils.force_array(expected)
        return expected,


@testing.parameterize(*testing.product({
    'function_name': ['argmax', 'argmin'],
}))
class TestArgMinMaxInvalid(unittest.TestCase):

    def setUp(self):
        self.function = getattr(functions, self.function_name)

        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            self.function(self.x, [0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            self.function(self.x, (1, 'x'))


testing.run_module(__name__, __file__)

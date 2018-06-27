import unittest

import numpy

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


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
class TestRepeat(unittest.TestCase):

    def setUp(self):
        self.in_shape = self.params['shape']
        self.repeats = self.params['repeats']
        self.axis = self.params['axis']
        self.x = numpy.random.uniform(-1, 1, self.in_shape).astype(self.dtype)
        out_shape = self._repeat(self.x, self.repeats, self.axis).shape
        self.gy = numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.in_shape) \
            .astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 2 ** -4, 'rtol': 2 ** -4}

    @classmethod
    def _repeat(cls, arr, repeats, axis=None):
        # Workaround NumPy 1.9 issue.
        if isinstance(repeats, tuple) and len(repeats) == 1:
            repeats = repeats[0]
        return numpy.repeat(arr, repeats, axis)

    def check_forward(self, x_data):
        y = functions.repeat(x_data, self.repeats, self.axis)
        y_expected = self._repeat(self.x, self.repeats, self.axis)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(
            y.data, y_expected, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.repeat(x, self.repeats, self.axis)

        gradient_check.check_backward(
            f, x_data, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        def f(x):
            return functions.repeat(x, self.repeats, self.axis)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad, **self.check_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                                   cuda.to_gpu(self.ggx))


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

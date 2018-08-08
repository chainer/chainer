import unittest

import numpy

from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (), 'pad_width': 1, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': 0, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': 1, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': (1, 2), 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': ((1, 2), (3, 4)), 'mode': 'constant'},
        {'shape': (2, 3, 2), 'pad_width': ((2, 5), (1, 2), (0, 7)),
         'mode': 'constant'},
        {'shape': (1, 3, 5, 2), 'pad_width': 2, 'mode': 'constant'}
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64}
    ]
))
class TestPadDefault(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_shape = numpy.pad(self.x, self.pad_width, self.mode).shape
        self.gy = numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 3e-2})

    def check_forward(self, x_data):
        y = functions.pad(x_data, self.pad_width, self.mode)
        y_expected = numpy.pad(self.x, self.pad_width, self.mode)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(y.data, y_expected)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, gy_data):
        def f(x):
            return functions.pad(x, pad_width=self.pad_width, mode=self.mode)

        gradient_check.check_backward(
            f, x_data, gy_data, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, gy_data, ggx_data):
        def f(x):
            return functions.pad(x, pad_width=self.pad_width, mode=self.mode)

        gradient_check.check_double_backward(
            f, x_data, gy_data, ggx_data, **self.check_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3), 'pad_width': 1, 'mode': 'constant',
         'constant_values': 1},
        {'shape': (2, 3), 'pad_width': (1, 2), 'mode': 'constant',
         'constant_values': (1, 2)},
        {'shape': (2, 3), 'pad_width': ((1, 2), (3, 4)), 'mode': 'constant',
         'constant_values': ((1, 2), (3, 4))},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64}
    ]
))
# Old numpy does not work with multi-dimensional constant_values
@testing.with_requires('numpy>=1.11.1')
class TestPad(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out_shape = numpy.pad(self.x, self.pad_width, mode=self.mode,
                              constant_values=self.constant_values).shape
        self.gy = numpy.random.uniform(-1, 1, out_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 3e-2})

    def check_forward(self, x_data):
        y = functions.pad(x_data, self.pad_width, mode=self.mode,
                          constant_values=self.constant_values)
        y_expected = numpy.pad(self.x, self.pad_width, mode=self.mode,
                               constant_values=self.constant_values)
        self.assertEqual(y.dtype, y_expected.dtype)
        testing.assert_allclose(y.data, y_expected)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, gy_data):
        def f(x):
            return functions.pad(
                x, pad_width=self.pad_width, mode=self.mode,
                constant_values=self.constant_values)

        gradient_check.check_backward(
            f, x_data, gy_data, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, gy_data, ggx_data):
        def f(x):
            return functions.pad(
                x, pad_width=self.pad_width, mode=self.mode,
                constant_values=self.constant_values)

        gradient_check.check_double_backward(
            f, x_data, gy_data, ggx_data, **self.check_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)

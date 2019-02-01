import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16},
     {'dtype': numpy.float32},
     {'dtype': numpy.float64},
     ],
    [{'slices': (0, slice(0, 1), numpy.array(-1)), 'b_data': numpy.array([1])},
     {'slices': (slice(None), 0, [0, 2]),
      'b_data': numpy.random.uniform(size=(4, 2))},
     {'slices': ([1, 0], [0, 0], [2, 0]),
      'b_data': numpy.random.uniform(size=(2,))},
     {'slices': 1, 'b_data': numpy.random.uniform(size=(2, 3))},
     {'slices': numpy.array([False, True, False, True]),
      'b_data': numpy.random.uniform(size=(2, 2, 3))},
     {'slices': [], 'b_data': numpy.empty(shape=(0, 2, 3))},
     ]
))
class TestScatterAdd(unittest.TestCase):

    def setUp(self):
        self.shape = (4, 2, 3)
        self.a_data = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)
        self.a_data_original = self.a_data.copy()
        self.gy_data = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)
        self.b_data = self.b_data.astype(self.dtype)
        self.gga_data = numpy.random.uniform(
            -1, 1, self.a_data.shape).astype(self.dtype)
        self.ggb_data = numpy.random.uniform(
            -1, 1, self.b_data.shape).astype(self.dtype)

        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-4}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_backward_options['dtype'] = numpy.float64
            self.check_double_backward_options['dtype'] = numpy.float64

    def check_forward(self, a_data, b_data):
        a = chainer.Variable(a_data)
        b = chainer.Variable(b_data)
        y = functions.scatter_add(a, self.slices, b)
        self.assertEqual(y.data.dtype, self.dtype)
        # Test to make sure that the input values are not changed
        numpy.testing.assert_equal(cuda.to_cpu(a.data), self.a_data_original)

        a_data_copy = cuda.to_cpu(a_data).copy()
        numpy.add.at(a_data_copy, self.slices, cuda.to_cpu(b_data))
        numpy.testing.assert_equal(a_data_copy, cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.a_data, self.b_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.a_data), cuda.to_gpu(self.b_data))

    def check_backward(self, a_data, b_data, y_grad):
        def f(a, b):
            return functions.scatter_add(a, self.slices, b)

        gradient_check.check_backward(
            f, (a_data, b_data), y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.a_data, self.b_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.a_data), cuda.to_gpu(self.b_data),
                            cuda.to_gpu(self.gy_data))

    def check_double_backward(self, a_data, b_data, y_grad, a_grad_grad,
                              b_grad_grad):
        def f(a, b):
            return functions.scatter_add(a, self.slices, b)

        gradient_check.check_double_backward(
            f, (a_data, b_data), y_grad, (a_grad_grad, b_grad_grad),
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.a_data, self.b_data, self.gy_data,
                                   self.gga_data, self.ggb_data)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.a_data),
                                   cuda.to_gpu(self.b_data),
                                   cuda.to_gpu(self.gy_data),
                                   cuda.to_gpu(self.gga_data),
                                   cuda.to_gpu(self.ggb_data))


class TestInvalidScatterAdd(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

        self.a_data = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.b_data = numpy.random.uniform(-1, 1, (2, 2))

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_multiple_ellipsis(self):
        with self.assertRaises(ValueError):
            functions.scatter_add(
                self.a_data, (Ellipsis, Ellipsis), self.b_data)

    def test_too_many_indices(self):
        with self.assertRaises(type_check.InvalidType):
            functions.scatter_add(self.a_data, (0, 0, 0, 0), self.b_data)

    def test_requires_broadcasting(self):
        with self.assertRaises(ValueError):
            functions.scatter_add(self.a_data, slice(0, 2), self.b_data)


testing.run_module(__name__, __file__)

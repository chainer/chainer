import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import parameterize
from chainer.utils import type_check
from chainer import utils


@parameterize(
    {'axes': [1, 2], 'offsets': 0,
     'value_data': numpy.random.uniform(-1, 1, (4, 2, 1))},
    {'axes': [1, 2], 'offsets': [0, 1, 1],
     'value_data': numpy.random.uniform(-1, 1, (4, 2, 1))},
    {'axes': 1, 'offsets': 1,
     'value_data': numpy.random.uniform(-1, 1, (4, 2, 2))},
    {'axes': 1, 'offsets': [0, 1, 1],
     'value_data': numpy.random.uniform(-1, 1, (4, 2, 2))},
    {'axes': [], 'offsets': 0, 'new_axes': 0,
     'value_data': numpy.random.uniform(-1, 1, (1, 4, 3, 2))},
    {'axes': [], 'offsets': 0, 'new_axes': 2,
     'value_data': numpy.random.uniform(-1, 1, (4, 3, 1, 2))},
    {'axes': [], 'offsets': 0, 'new_axes': 3,
     'value_data': numpy.random.uniform(-1, 1, (4, 3, 2, 1))},
    {'slices': (1, -1),
     'value_data': numpy.arange(2, dtype=numpy.float64)},
    {'slices': (1, Ellipsis, -1),
     'value_data': numpy.arange(3, dtype=numpy.float64)},
    {'slices': (1, None, Ellipsis, None, -1),
     'value_data': numpy.arange(3, dtype=numpy.float64).reshape(1, 3, 1)},
    {'slices': (1, -1, 0),
     'value_data': numpy.array(1, dtype=numpy.float64)},
    {'slices': (0, 0, 0),
     'value_data': numpy.array(1, dtype=numpy.float64)},
)
class TestSetItem(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.shape = (4, 2, 1)

        if not hasattr(self, 'slices'):
            # Convert axes, offsets and shape to slices
            if isinstance(self.offsets, int):
                self.offsets = tuple([self.offsets] * len(self.shape))
            if isinstance(self.axes, int):
                self.axes = tuple([self.axes])

            self.slices = [slice(None)] * len(self.shape)
            for axis in self.axes:
                self.slices[axis] = slice(
                    self.offsets[axis], self.offsets[axis] + self.shape[axis])

            if hasattr(self, 'new_axes'):
                self.slices.insert(self.new_axes, None)

            self.slices = tuple(self.slices)

        self.gy_data = numpy.zeros_like(self.x_data)
        self.gy_data[self.slices] = \
            numpy.random.uniform(-1, 1, self.x_data.shape)[self.slices]

    def check_forward(self, x_data, value_data):
        x = chainer.Variable(x_data)
        value = chainer.Variable(value_data)
        functions.set_item(x, self.slices, value)
        numpy.testing.assert_equal(cuda.to_cpu(value_data),
                                   cuda.to_cpu(x.data)[self.slices])

    def test_forward_cpu(self):
        self.check_forward(self.x_data, self.value_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.value_data))

    def check_backward(self, x_data, value_data, y_grad):
        # gradient with y
        x = chainer.Variable(x_data)
        value = chainer.Variable(value_data)
        y, _ = functions.set_item(x, self.slices, value)
        y.grad = y_grad
        y.backward()
        testing.assert_allclose(x.grad, y_grad)
        testing.assert_allclose(value.grad, y_grad[self.slices])

        # gradient with yvalue
        x = chainer.Variable(x_data)
        value = chainer.Variable(value_data)
        y, yvalue = functions.set_item(x, self.slices, value)
        y.grad = y_grad
        yvalue.grad = utils.force_array(y_grad[self.slices])
        yvalue.backward()
        testing.assert_allclose(x.grad, y_grad)
        testing.assert_allclose(value.grad, y_grad[self.slices])

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.value_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.value_data),
                            cuda.to_gpu(self.gy_data))


class TestInvalidSetItem(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2))

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_advanced_indexing(self):
        with self.assertRaises(ValueError):
            value = numpy.arange(3 * 3 * 2).reshape(3, 3, 2)
            value = value.astype(numpy.float64)
            functions.set_item(self.x_data, ([0, 0, 0],), value)

    def test_multiple_ellipsis(self):
        with self.assertRaises(ValueError):
            value = numpy.arange(4 * 3 * 2).reshape(4, 3, 2)
            value = value.astype(numpy.float64)
            functions.set_item(self.x_data, (Ellipsis, Ellipsis), value)

    def test_too_many_indices(self):
        with self.assertRaises(type_check.InvalidType):
            value = numpy.zeros(1, dtype=numpy.float64)
            functions.set_item(self.x_data, (0, 0, 0, 0), value)


testing.run_module(__name__, __file__)

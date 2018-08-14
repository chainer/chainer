import sys
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import parameterize


@parameterize(*(testing.product(
    {
        'shape': [(4, 3, 2)],
        'batch_ndim': [0],
        'slices': [
            # from test_get_item.TestGetItem
            (1, -1, 0),
            (1, -1),
            (1, Ellipsis, -1),
            (1, None, Ellipsis, None, -1),

            # from test_get_item.TestGetItemAdvanced
            [],
            ([],),
            ([[]],),
            numpy.array([], dtype=numpy.bool),
            (1, [1]),
            ([1], slice(1, 2)),
            [1, 0],
            ([1, 0],),
            numpy.array([[1, 0], [2, 3]]),
            ([1, 0], [1, 1]),
            ([1, 0], slice(None), [[1, 1], [0, 0]]),
            ([1, 0], slice(1, 2), [0, 0]),
            ([[1, 2], [3, 0]], slice(1, 2), 1),
            numpy.array([True] * 18 + [False] * 6).reshape(4, 3, 2),
            numpy.array([True, False, False, True]),
            (slice(None), numpy.array([True, False, True])),
            numpy.array([False, False, False, False]),
            (3, 2, Ellipsis, 1),
            (numpy.array(False)),
            (numpy.array(True)),
        ],
        'debug': [False],
    },
) + testing.product_dict([
    {'shape': (4, 3, 2), 'slices': (1, -1), 'batch_ndim': 1},
    {'shape': (4, 3, 2), 'slices': (Ellipsis, 1), 'batch_ndim': 2},
    {'shape': (), 'slices': (), 'batch_ndim': 0},
    {'shape': (), 'slices': None, 'batch_ndim': 0},
    {'shape': (), 'slices': None, 'batch_ndim': 1},
], [
    {'debug': False},
    {'debug': True},
])))
class TestCopiedSetItem(unittest.TestCase):

    def setUp(self):
        self.x0_data = numpy.random.uniform(-1, 1, self.shape)
        try:
            sliced_shape = self.x0_data[self.slices].shape
        except IndexError as e:
            self.skipTest(
                "not supported in this version of numpy ({})".format(e))
        assert 0 <= self.batch_ndim <= len(sliced_shape)
        rhs_shape = sliced_shape[self.batch_ndim:]
        self.x1_data = numpy.random.uniform(-1, 1, rhs_shape)
        self.gy_data = numpy.random.uniform(-1, 1, self.shape)
        self.ggx0_data = numpy.random.uniform(-1, 1, self.shape)
        self.ggx1_data = numpy.random.uniform(-1, 1, rhs_shape)
        self.ctx = chainer.using_config('debug', self.debug)
        self.ctx.__enter__()

    def tearDown(self):
        self.ctx.__exit__(*sys.exc_info())

    def _forward(self, x0, x1):
        return functions.copied_set_item(x0, self.slices, x1)

    def check_forward(self, x0_data, x1_data):
        y_expected = x0_data.copy()
        y_expected[self.slices] = x1_data

        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        y = self._forward(x0, x1)
        testing.assert_allclose(y.array, y_expected)

    def test_forward_cpu(self):
        self.check_forward(self.x0_data, self.x1_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x0_data))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self._forward, x_data, y_grad)

    def test_backward_cpu(self):
        self.check_backward((self.x0_data, self.x1_data), self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            (cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x1_data)),
            cuda.to_gpu(self.gy_data))

    def check_double_backward(self, x_data, y_grad, ggx_data):
        gradient_check.check_double_backward(
            self._forward, x_data, y_grad, ggx_data)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            (self.x0_data, self.x1_data), self.gy_data,
            (self.ggx0_data, self.ggx1_data))

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            (cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x1_data)),
            cuda.to_gpu(self.gy_data),
            (cuda.to_gpu(self.ggx0_data), cuda.to_gpu(self.ggx1_data)))


@parameterize(
    {'shape': (1,), 'slices': (
        numpy.array([0, 0]),
    ), 'batch_ndim': 0},
    {'shape': (3, 2), 'slices': (
        numpy.array([1, 0, 1, 2]), numpy.array([0, 1, 0, 1])
    ), 'batch_ndim': 0},
    {'shape': (3, 2), 'slices': (
        numpy.array([1, 0, 1, 2]), numpy.array([0, 1, 0, 1])
    ), 'batch_ndim': 1},
    {'shape': (3, 2), 'slices': (
        numpy.array([[1, 0], [2, 1]]), numpy.array([[0, 1], [1, 0]])
    ), 'batch_ndim': 0},
    {'shape': (3, 2), 'slices': (
        numpy.array([[1, 0], [2, 1]]), numpy.array([[0, 1], [1, 0]])
    ), 'batch_ndim': 2},
    {'shape': (3, 2), 'slices': (
        numpy.array([[1, 0], [1, 2]]), numpy.array([[0, 1]])
    ), 'batch_ndim': 0},
    {'shape': (3, 2), 'slices': (
        numpy.array([[1, 0], [1, 2]]), numpy.array([[0, 1]])
    ), 'batch_ndim': 2},
)
class TestCopiedSetItemRaise(unittest.TestCase):

    def setUp(self):
        self.x0_data = numpy.random.uniform(-1, 1, self.shape)
        try:
            sliced_shape = self.x0_data[self.slices].shape
        except IndexError as e:
            self.skipTest(
                "not supported in this version of numpy ({})".format(e))
        assert 0 <= self.batch_ndim <= len(sliced_shape)
        rhs_shape = sliced_shape[self.batch_ndim:]
        self.x1_data = numpy.random.uniform(-1, 1, rhs_shape)
        self.gy_data = numpy.random.uniform(-1, 1, self.shape)

    def check_backward_raise(self, x_data, gy_data):
        x0_data, x1_data = x_data
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        y = functions.copied_set_item(x0, self.slices, x1)
        y.grad = gy_data
        with chainer.using_config('debug', True):
            with self.assertRaises(ValueError):
                y.backward()

    def test_backward_cpu(self):
        self.check_backward_raise((self.x0_data, self.x1_data), self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward_raise(
            (cuda.to_gpu(self.x0_data), cuda.to_gpu(self.x1_data)),
            cuda.to_gpu(self.gy_data))


testing.run_module(__name__, __file__)

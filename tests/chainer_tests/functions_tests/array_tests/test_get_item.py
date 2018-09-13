import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16},
     {'dtype': numpy.float32},
     {'dtype': numpy.float64},
     ],
    [{'axes': [1, 2], 'offsets': 0, 'sliced_shape': (4, 2, 1)},
     {'axes': [1, 2], 'offsets': [0, 1, 1], 'sliced_shape': (4, 2, 1)},
     {'axes': 1, 'offsets': 1, 'sliced_shape': (4, 2, 2)},
     {'axes': 1, 'offsets': [0, 1, 1], 'sliced_shape': (4, 2, 2)},
     {'axes': [], 'offsets': 0, 'new_axes': 0, 'sliced_shape': (1, 4, 3, 2)},
     {'axes': [], 'offsets': 0, 'new_axes': 2, 'sliced_shape': (4, 3, 1, 2)},
     {'axes': [], 'offsets': 0, 'new_axes': 3, 'sliced_shape': (4, 3, 2, 1)},
     {'slices': (1, -1, 0), 'sliced_shape': ()},
     {'slices': (1, -1), 'sliced_shape': (2,)},
     {'slices': (1, Ellipsis, -1), 'sliced_shape': (3,)},
     {'slices': (1, None, Ellipsis, None, -1), 'sliced_shape': (1, 3, 1)},
    ]
))
class TestGetItem(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(self.dtype)
        self.shape = (4, 2, 1)
        self.gy_data = numpy.random.uniform(
            -1, 1, self.sliced_shape).astype(self.dtype)
        self.ggx_data = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(self.dtype)

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

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.get_item(x, self.slices)
        self.assertEqual(y.data.dtype, self.dtype)
        numpy.testing.assert_equal(cuda.to_cpu(x_data)[self.slices],
                                   cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.get_item(x, self.slices)

        gradient_check.check_backward(
            f, (x_data,), y_grad, dtype='d')

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.gy_data))

    def check_double_backward(self, x_data, y_grad, ggx_data):
        def f(x):
            return functions.get_item(x, self.slices)

        gradient_check.check_double_backward(
            f, (x_data,), y_grad, ggx_data, dtype='d')

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x_data, self.gy_data, self.ggx_data)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x_data),
                                   cuda.to_gpu(self.gy_data),
                                   cuda.to_gpu(self.ggx_data))


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16},
     {'dtype': numpy.float32},
     {'dtype': numpy.float64},
     ],
    [{'slices': [], 'sliced_shape': (0, 3, 2)},
     {'slices': ([],), 'sliced_shape': (0, 3, 2)},
     {'slices': ([[]],), 'sliced_shape': (1, 0, 3, 2)},
     {'slices': numpy.array([], dtype=numpy.bool),
         'sliced_shape': (0, 3, 2)},
     {'slices': (1, [1]), 'sliced_shape': (1, 2)},
     {'slices': ([1], slice(1, 2)), 'sliced_shape': (1, 1, 2)},
     {'slices': [1, 0], 'sliced_shape': (2, 3, 2)},
     {'slices': ([1, 0],), 'sliced_shape': (2, 3, 2)},
     {'slices': numpy.array([[1, 0], [2, 3]]),
         'sliced_shape': (2, 2, 3, 2)},
     {'slices': ([1, 0], [1, 1]), 'sliced_shape': (2, 2)},
     {'slices': ([1, 0], slice(None), [[1, 1], [1, 1]]),
         'sliced_shape': (2, 2, 3)},
     {'slices': ([1, 0], slice(1, 2), [0, 0]), 'sliced_shape': (2, 1)},
     {'slices': ([[1, 1], [1, 0]], slice(1, 2), 1),
         'sliced_shape': (2, 2, 1)},
     {'slices': numpy.array([True] * 18 + [False] * 6).reshape(4, 3, 2),
         'sliced_shape': (18,)},
     {'slices': numpy.array([True, False, False, True]),
         'sliced_shape': (2, 3, 2)},
     {'slices': (slice(None), numpy.array([True, False, True])),
         'sliced_shape': (4, 2, 2)},
     {'slices': numpy.array([False, False, False, False]),
         'sliced_shape': (0, 3, 2)},
     {'slices': (3, 2, Ellipsis, 1),
         'sliced_shape': ()},
     {'slices': (numpy.array(False)),
         'input_shape': (), 'sliced_shape': (0,)},
     {'slices': (numpy.array(True)),
         'input_shape': (), 'sliced_shape': (1,)},
     ]
))
class TestGetItemAdvanced(unittest.TestCase):

    input_shape = (4, 3, 2)

    def setUp(self):
        self.x_data = numpy.random.uniform(
            -1, 1, self.input_shape).astype(self.dtype)
        self.gy_data = numpy.random.uniform(
            -1, 1, self.sliced_shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.get_item(x, self.slices)
        self.assertEqual(y.data.dtype, self.dtype)
        numpy.testing.assert_equal(cuda.to_cpu(x_data)[self.slices],
                                   cuda.to_cpu(y.data))

    def test_forward_cpu(self):
        self.check_forward(self.x_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.get_item(x, self.slices)

        gradient_check.check_backward(
            f, (x_data,), y_grad, dtype='d')

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.gy_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.gy_data))


@testing.parameterize(
    {'slices': ([1, 0], [1, 1]), 'sliced_shape': (2, 2)},
    {'slices': ([1, 0], slice(None), [[1, 1], [1, 1]]),
     'sliced_shape': (2, 2, 3)},
    {'slices': ([1, 0], [1, 1], [0, 0]), 'sliced_shape': (2,)},
    {'slices': (slice(None), numpy.array([True, False, True])),
     'sliced_shape': (4, 2, 2)},
)
class TestCupyIndicesGetItem(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(
            -1, 1, (4, 3, 2)).astype(numpy.float32)
        self.gy_data = numpy.random.uniform(
            -1, 1, self.sliced_shape).astype(numpy.float32)

    def check_forward(self, x_data):
        slices = []
        for i, s in enumerate(self.slices):
            if isinstance(s, numpy.ndarray):
                s = chainer.backends.cuda.cupy.array(s)
            if isinstance(s, list):
                s = chainer.backends.cuda.cupy.array(s, dtype=numpy.int32)
            slices.append(s)
        slices = tuple(slices)
        x = chainer.Variable(x_data)
        y = functions.get_item(x, slices)
        self.assertEqual(y.data.dtype, numpy.float32)
        numpy.testing.assert_equal(cuda.to_cpu(x_data)[self.slices],
                                   cuda.to_cpu(y.data))

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data))

    def check_backward(self, x_data, y_grad):
        slices = []
        for i, s in enumerate(self.slices):
            if isinstance(s, numpy.ndarray):
                s = chainer.backends.cuda.cupy.array(s)
            if isinstance(s, list):
                s = chainer.backends.cuda.cupy.array(s, dtype=numpy.int32)
            slices.append(s)
        slices = tuple(slices)

        def f(x):
            return functions.get_item(x, slices)

        gradient_check.check_backward(
            f, (x_data,), y_grad, dtype='d')

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.gy_data))


class TestInvalidGetItem(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

        self.x_data = numpy.random.uniform(-1, 1, (4, 3, 2))

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_multiple_ellipsis(self):
        with self.assertRaises(ValueError):
            functions.get_item(self.x_data, (Ellipsis, Ellipsis))


testing.run_module(__name__, __file__)

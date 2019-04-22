import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


_backend_params = (
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + [{'use_cuda': True}]
    # ChainerX tests
    + [
        # TODO(niboshi): Add the following configurations
        # {'use_chainerx': True, 'chainerx_device': 'native:0'},
        # {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        # {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])


@testing.inject_backend_tests(None, _backend_params)
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
class TestGetItem(testing.FunctionTestCase):

    def setUp(self):
        shape = (4, 2, 1)

        if not hasattr(self, 'slices'):
            axes = self.axes
            offsets = self.offsets

            # Convert axes, offsets and shape to slices
            if isinstance(offsets, int):
                offsets = tuple([offsets] * len(shape))
            if isinstance(axes, int):
                axes = tuple([axes])

            slices = [slice(None)] * len(shape)
            for axis in axes:
                slices[axis] = slice(
                    offsets[axis], offsets[axis] + shape[axis])

            if hasattr(self, 'new_axes'):
                slices.insert(self.new_axes, None)

            self.axes = axes
            self.offsets = offsets
            self.slices = tuple(slices)

        self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-4})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.get_item(x, self.slices)
        assert y.shape == self.sliced_shape
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        y = x[self.slices]
        return numpy.asarray(y),


@testing.inject_backend_tests(None, _backend_params)
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
class TestGetItemAdvanced(testing.FunctionTestCase):

    input_shape = (4, 3, 2)

    def setUp(self):
        self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-4})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.input_shape).astype(self.dtype)
        return x,

    def _convert_slices(self, slices, device):
        # Converts advanced indexing slices (of numpy.ndarray) to respective
        # backend arrays.
        if isinstance(slices, list):
            return [self._convert_slices(a, device) for a in slices]
        if isinstance(slices, tuple):
            return tuple([self._convert_slices(a, device) for a in slices])
        if isinstance(slices, numpy.ndarray):
            return device.send(slices)
        return slices

    def forward(self, inputs, device):
        x, = inputs
        slices = self._convert_slices(self.slices, device)
        y = functions.get_item(x, slices)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        y = x[self.slices]
        return numpy.asarray(y),


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

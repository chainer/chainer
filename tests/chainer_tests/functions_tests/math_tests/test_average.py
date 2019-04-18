import unittest

import numpy
import six

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*(
    testing.product({
        'shape': [(3, 2, 4)],
        'axis': [None, 0, 1, 2, -1, (0, 1), (1, -1)],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'use_weights': [True, False],
        'keepdims': [True, False],
    }) +
    testing.product({
        'shape': [()],
        'axis': [None],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'use_weights': [True, False],
        'keepdims': [True, False],
    })))
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
    }))
class TestAverage(testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = True

        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 5e-3, 'rtol': 5e-3})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-1})
        else:
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})

    def before_test(self, test_name):
        if self.use_weights and isinstance(self.axis, tuple):
            # This condition is not supported
            raise unittest.SkipTest(
                'Tuple axis is not supported when weights is given')

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        if self.axis is None:
            w_shape = self.shape
        elif isinstance(self.axis, int):
            axis = self.axis
            if axis < 0:
                ndim = len(self.shape)
                axis += ndim
            w_shape = self.shape[axis],
        else:
            w_shape = tuple(self.shape[a] for a in self.axis)

        # Sample weights. Weights should not sum to 0.
        while True:
            w = numpy.random.uniform(-2, 2, w_shape).astype(self.dtype)
            w_sum_eps = 1.0 if self.dtype == numpy.float16 else 5e-2
            if abs(w.sum()) > w_sum_eps:
                break

        return x, w

    def forward(self, inputs, device):
        x, w = inputs
        if not self.use_weights:
            w = None
        y = functions.average(
            x, axis=self.axis, weights=w, keepdims=self.keepdims)
        return y,

    def forward_expected(self, inputs):
        x, w = inputs
        if not self.use_weights:
            w = None
        y_expect = numpy.average(x, axis=self.axis, weights=w)
        if self.keepdims:
            # numpy.average does not support keepdims
            axis = self.axis
            if axis is None:
                axis = list(six.moves.range(x.ndim))
            elif isinstance(axis, int):
                axis = axis,
            shape = list(x.shape)
            for i in six.moves.range(len(shape)):
                if i in axis or i - len(shape) in axis:
                    shape[i] = 1
            y_expect = y_expect.reshape(shape)
        y_expect = utils.force_array(y_expect, dtype=self.dtype)
        return y_expect,


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestAverageDuplicateValueInAxis(unittest.TestCase):

    def test_duplicate_value(self):
        x = numpy.random.uniform(-1, 1, 24).reshape(2, 3, 4).astype(self.dtype)
        with self.assertRaises(ValueError):
            functions.average(x, axis=(0, 0))

    def test_duplicate_value_negative(self):
        x = numpy.random.uniform(-1, 1, 24).reshape(2, 3, 4).astype(self.dtype)
        with self.assertRaises(ValueError):
            functions.average(x, axis=(1, -2))

    def test_weights_and_axis(self):
        x = numpy.random.uniform(-1, 1, 24).reshape(2, 3, 4).astype(self.dtype)
        w = numpy.random.uniform(-1, 1, 6).reshape(2, 3).astype(self.dtype)
        with self.assertRaises(ValueError):
            functions.average(x, axis=(0, 1), weights=w)


testing.run_module(__name__, __file__)

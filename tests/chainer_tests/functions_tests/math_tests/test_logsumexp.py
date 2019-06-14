import unittest

import numpy

from chainer import functions
from chainer import testing
from chainer.utils import force_array


@testing.parameterize(
    *testing.product({
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
        'shape': [(), (3, 2, 4)],
        'axis': [None, 0, 1, 2, -1, (0, 1), (1, 0), (0, -1), (-2, 0)]
    })
)
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
class TestLogSumExp(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'eps': 2.0 ** -3, 'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'eps': 2.0 ** -3, 'rtol': 1e-1, 'atol': 1e-1})
        else:
            self.check_backward_options.update({
                'eps': 2.0 ** -5, 'rtol': 1e-4, 'atol': 1e-4})
            self.check_double_backward_options.update({
                'eps': 2.0 ** -5, 'rtol': 1e-4, 'atol': 1e-4})

    def before_test(self, test_name):
        if self.axis is not None and self.shape == ():
            raise unittest.SkipTest('Axis must be None on 0-dim input.')

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.logsumexp(x, axis=self.axis),

    def forward_expected(self, inputs):
        x, = inputs
        expected = numpy.log(numpy.exp(x).sum(axis=self.axis))
        expected = force_array(expected)
        return expected,


class TestLogSumExpInvalidAxis(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.logsumexp(self.x, [0])

    def test_invalid_axis_type_in_tuple(self):
        with self.assertRaises(TypeError):
            functions.logsumexp(self.x, (1, 'x'))

    def test_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.logsumexp(self.x, (0, 0))

    def test_pos_neg_duplicate_axis(self):
        with self.assertRaises(ValueError):
            functions.logsumexp(self.x, (1, -2))


testing.run_module(__name__, __file__)

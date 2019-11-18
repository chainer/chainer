import unittest

import numpy as np
import six

import chainer
from chainer import functions
from chainer import testing
from chainer.utils import type_check


def _as_two_dim(x):
    if x.ndim == 2:
        return x
    return x.reshape((len(x), -1))


@testing.parameterize(*testing.product({
    'dtype': [np.float16, np.float32, np.float64],
    'shape': [(4, 3, 5), (4, 15)],
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
    }))
class TestBatchL2NormSquared(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == np.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update({
                'atol': 1e-2, 'rtol': 1e-2})

    def generate_inputs(self):
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.batch_l2_norm_squared(x),

    def forward_expected(self, inputs):
        x, = inputs
        x_two_dim = _as_two_dim(x)
        y_expect = np.empty(len(x), dtype=self.dtype)
        for n in six.moves.range(len(x)):
            y_expect[n] = sum(map(lambda x: x * x, x_two_dim[n]))
        return y_expect,


class TestBatchL2NormSquaredTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x = chainer.Variable(np.zeros((4,), dtype=np.float32))

        with self.assertRaises(type_check.InvalidType):
            functions.batch_l2_norm_squared(x)


testing.run_module(__name__, __file__)

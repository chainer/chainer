import numpy
import six

from chainer import functions
from chainer import testing
from chainer.testing import backend


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@backend.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestLocalResponseNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = True
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-3}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {'atol': 3e-4, 'rtol': 3e-3}

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, (2, 7, 3, 2)).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        # Naive implementation
        x, = inputs
        y_expect = numpy.zeros_like(x)
        for n, c, h, w in numpy.ndindex(x.shape):
            s = 0
            for i in six.moves.range(max(0, c - 2), min(7, c + 2)):
                s += x[n, i, h, w] ** 2
            denom = (2 + 1e-4 * s) ** .75
            y_expect[n, c, h, w] = x[n, c, h, w] / denom
        return y_expect,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.local_response_normalization(x)
        return y,


testing.run_module(__name__, __file__)

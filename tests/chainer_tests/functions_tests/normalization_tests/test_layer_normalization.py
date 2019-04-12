import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*(testing.product({
    'batchsize': [1, 5],
    'size': [10, 20],
    'dtype': [numpy.float32],
    'eps': [1e-5, 1e-1],
})))
@testing.inject_backend_tests(
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
    ]
)
class TestLayerNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def generate_inputs(self):
        shape = self.batchsize, self.size
        size = numpy.prod(shape) // shape[0]
        x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        gamma = numpy.random.uniform(-1, 1, size).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, size).astype(self.dtype)
        return x, gamma, beta

    def forward_expected(self, inputs):
        x, gamma, beta = inputs
        mean = numpy.mean(x, axis=1, keepdims=True)
        var = numpy.mean(numpy.square(x - mean), axis=1, keepdims=True)
        std = numpy.sqrt(var + self.eps)
        y_expected = (
            numpy.expand_dims(gamma, axis=0) * (x - mean) / std
            + numpy.expand_dims(beta, axis=0))
        return y_expected,

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        y = functions.layer_normalization(x, gamma, beta, eps=self.eps)
        return y,


testing.run_module(__name__, __file__)

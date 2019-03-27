import six

import numpy

from chainer import functions
import chainer.functions.normalization.group_normalization as gn_module
from chainer import testing


def _simple_group_normalization(x, groups, gamma, beta, eps=1e-5):
    batch_size, channels = x.shape[:2]
    x_reshape = x.reshape(batch_size, groups, channels // groups, -1)

    mean = numpy.mean(x_reshape, axis=(2, 3), keepdims=True)
    var = numpy.var(x_reshape, axis=(2, 3), keepdims=True)
    std = numpy.sqrt(var + eps, dtype=x.dtype)

    x_hat = (x_reshape - mean) / std
    x_hat = x_hat.reshape(x.shape)

    for i in six.moves.xrange(x.ndim):
        if i != 1:  # except for channel dim
            gamma = numpy.expand_dims(gamma, i)
            beta = numpy.expand_dims(beta, i)

    return x_hat * gamma + beta


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'groups': [1, 2, 4],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'eps': [1e-5, 1e-1],
})))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestGroupNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
        self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 1e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gamma = numpy.random.uniform(-1, 1, self.shape[1]).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, self.shape[1]).astype(self.dtype)
        return x, gamma, beta

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        y = functions.group_normalization(x, self.groups, gamma, beta,
                                          eps=self.eps)
        return y,

    def forward_expected(self, inputs):
        x, gamma, beta = inputs
        y = _simple_group_normalization(x, self.groups, gamma, beta,
                                        eps=self.eps)
        return y,


@testing.parameterize(*(testing.product({
    'shape': [(15, 10)],
    'dtype': [numpy.float32],
    'eps': [1e-5, 1e-1],
})))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestMulInvStd(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        y = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, y

    def forward(self, inputs, device):
        x, y = inputs

        mean = functions.mean(x, axis=1)
        d = x - mean[:, None]
        var = functions.mean(d * d, axis=1)
        inv_std = functions.rsqrt(var + self.eps)

        dummy_gamma = self.backend_config.xp.ones(
            self.shape[0], dtype=self.dtype)

        return gn_module._MulInvStd(
            self.eps, mean.array, inv_std.array, dummy_gamma).apply((x, y))

    def forward_expected(self, inputs):
        x, y = inputs
        inv_std = (numpy.var(x, axis=1) + self.eps) ** -0.5
        z = inv_std[:, None] * y
        return z,


testing.run_module(__name__, __file__)

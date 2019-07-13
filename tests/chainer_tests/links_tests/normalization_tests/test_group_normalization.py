import six
import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr
import chainerx


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


class GroupNormalizationTestBase(object):

    param_names = ('gamma', 'beta')

    def setUp(self):
        if self.dtype == chainer.mixed16:
            self.highprec_dtype = numpy.float32
        else:
            self.highprec_dtype = self.dtype

        self.eps = 1e-5
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype in (numpy.float16, chainer.mixed16):
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-1}
            self.check_backward_options = {'atol': 5e-1, 'rtol': 1e-1}

    def before_test(self, test_name):
        if (self.dtype == chainer.mixed16
                and self.backend_config.xp is chainerx):
            raise unittest.SkipTest(
                'ChainerX does not yet support mixed-FP16 mode.')

    def generate_params(self):
        initial_gamma = numpy.random.uniform(
            -1, 1, (self.groups,)).astype(self.highprec_dtype)
        initial_beta = numpy.random.uniform(
            -1, 1, (self.groups,)).astype(self.highprec_dtype)
        return initial_gamma, initial_beta

    def create_link(self, initializers):
        initial_gamma, initial_beta = initializers

        link = links.GroupNormalization(
            groups=self.groups,
            initial_gamma=initial_gamma, initial_beta=initial_beta,
            eps=self.eps
        )
        return link

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, link, inputs, device):
        x, = inputs
        y = link(x)
        return y,

    def forward_expected(self, link, inputs):
        gamma = link.gamma.array
        beta = link.beta.array
        x, = inputs

        y = _simple_group_normalization(x, self.groups, gamma, beta, self.eps)

        y_one_each = None
        if len(x) > 1:
            y_one_each = chainer.functions.concat(
                [self.link(numpy.expand_dims(one_x, axis=0))
                 for one_x in x_data], axis=0).array

        return y.astype(self.dtype), y_one_each

    def check_forward_outputs(self, outputs, expected_outputs):
        _, no_bs_effect = expected_outputs
        super(GroupNormalizationTestBase, self).check_forward_outputs(
            outputs, expected_outputs)
        y, = outputs
        assert y.dtype == chainer.get_dtype(self.dtype)
        if no_bs_effect is not None:
            testing.assert_allclose(
                y.array, no_bs_effect, **self.check_forward_options)


@testing.parameterize(*testing.product({
    'shape': [(1, 4, 5, 3), (5, 4, 7), (3, 20)],
    'groups': [1, 2, 4],
    'dtype': [numpy.float16, numpy.float32, numpy.float64, chainer.mixed16],
}))
@testing.inject_backend_tests(
    None,
    [
        {}, {'use_ideep': 'always'},
    ]
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
        'use_cudnn': ['never', 'always'],
    })
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class GroupNormalizationTest(GroupNormalizationTestBase, testing.LinkTestCase):

    pass


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3]
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.initial_gamma = numpy.random.uniform(-1, 1, self.size)
        self.initial_gamma = self.initial_gamma.astype(numpy.float32)
        self.initial_beta = numpy.random.uniform(-1, 1, self.size)
        self.initial_beta = self.initial_beta.astype(numpy.float32)
        self.link = links.GroupNormalization(self.groups,
                                             initial_gamma=self.initial_gamma,
                                             initial_beta=self.initial_beta)
        self.shape = (1, self.size, 1)

    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.array)
        testing.assert_allclose(self.initial_beta, self.link.beta.array)

    @attr.gpu
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype='f'))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.array)
        testing.assert_allclose(self.initial_beta, self.link.beta.array)


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3],
    'dtype': [numpy.float32],
}))
@testing.inject_backend_tests(
    ['test_initialization'],
    [
        {},
        {'use_ideep': 'always'},
        {'use_cuda': True},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ]
)
class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.shape = (1, self.size, 1)
        self.link = links.GroupNormalization(self.groups)
        self.x = numpy.ones(self.shape, dtype=self.dtype)

    def test_initialization(self, backend_config):
        x = backend_config.get_array(self.x)
        link = self.link.to_device(backend_config.device)
        with backend_config:
            link(x)
        testing.assert_allclose(self.x, self.link.gamma.array)
        testing.assert_allclose(
            numpy.zeros(self.shape, self.dtype), self.link.beta.array)


@testing.parameterize(*testing.product({
    'shape': [(2,), ()],
    'groups': [3],
    'dtype': [numpy.float32],
}))
@testing.inject_backend_tests(
    ['test_ivnvalid_shape'],
    [
        {},
        {'use_ideep': 'always'},
        {'use_cuda': True},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ]
)
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.link = links.GroupNormalization(groups=self.groups)
        self.x = numpy.zeros(self.shape, dtype=self.dtype)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            self.link.to_device(
                backend_config.device
            )(backend_config.get_array(self.x))


@testing.parameterize(*testing.product({
    'shape': [(2, 5, 2)],
    'dtype': [numpy.float32],
}))
@testing.inject_backend_tests(
    ['test_invalid_groups', 'test_invalid_type_groups'],
    [
        {},
        {'use_ideep': True},
        {'use_cuda': True},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ]
)
class TestInvalidInitialize(unittest.TestCase):

    def setUp(self):
        shape = (2, 5, 2)
        self.x = chainer.Variable(numpy.zeros(shape, dtype='f'))

    def test_invalid_groups(self, backend_config):
        self.link = links.GroupNormalization(groups=3)
        with pytest.raises(ValueError):
            with backend_config:
                self.link.to_device(
                    backend_config.device
                )(backend_config.get_array(self.x))

    def test_invalid_type_groups(self):
        self.link = links.GroupNormalization(groups=3.5)
        with pytest.raises(TypeError):
            self.link.to_device(
                backend_config.device
            )(backend_config.get_array(self.x))


testing.run_module(__name__, __file__)

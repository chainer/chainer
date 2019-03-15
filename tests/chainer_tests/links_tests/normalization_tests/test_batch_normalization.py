import unittest

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import gradient_check
from chainer import initializers
from chainer import links
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check
import chainerx


_parameterize = testing.parameterize(*(testing.product_dict(
    testing.product({
        'test': [True, False],
        'size': ['skip', 'explicit'],
        'dtype': [numpy.float16, numpy.float32, numpy.float64,
                  chainer.mixed16],
    }),
    testing.product({
        'ndim': [0, 1, 2, 3],
    }) + [
        {'input_shape': (5, 4, 3, 2), 'axis': (0, 2, 3)},
        {'input_shape': (5, 4), 'axis': 0},
        {'input_shape': (5, 4, 3), 'axis': (0, 1)},
    ]
)))


_inject_backend_tests = testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
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


def _batch_normalization(expander, gamma, beta, x, mean, var, eps, test):
    mean = mean[expander]
    if test:
        std = numpy.sqrt(var[expander])
    else:
        std = numpy.sqrt(var[expander] + eps)
    y_expect = gamma * (x - mean) / std + beta
    return y_expect


class BatchNormalizationTestBase(object):

    param_names = ('gamma', 'beta')

    def setUp(self):
        if self.dtype == chainer.mixed16:
            self.highprec_dtype = numpy.float32
        else:
            self.highprec_dtype = self.dtype

        if hasattr(self, 'axis') and hasattr(self, 'input_shape'):
            aggr_axes = self.axis
            if isinstance(aggr_axes, int):
                aggr_axes = aggr_axes,
            shape = self.input_shape
            param_shape = tuple(
                s for i, s in enumerate(shape) if i not in aggr_axes)
            expander = tuple(
                None if i in aggr_axes else slice(None)
                for i in range(len(shape)))
        elif hasattr(self, 'ndim'):
            aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
            shape = (5, 3) + (2,) * self.ndim
            param_shape = shape[1]
            expander = (None, Ellipsis) + (None,) * self.ndim
        else:
            assert False

        self.aggr_axes = aggr_axes
        self.shape = shape
        self.param_shape = param_shape
        self.expander = expander
        self.finetune = False
        self.eps = 2e-5

        if self.test:
            self.mean = numpy.random.uniform(
                -1, 1, param_shape).astype(self.highprec_dtype)
            self.var = numpy.random.uniform(
                0.5, 1, param_shape).astype(self.highprec_dtype)
        else:
            self.mean = None
            self.var = None

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
            -1, 1, self.param_shape).astype(self.highprec_dtype)
        initial_beta = numpy.random.uniform(
            -1, 1, self.param_shape).astype(self.highprec_dtype)
        return initial_gamma, initial_beta

    def create_link(self, initializers):
        initial_gamma, initial_beta = initializers

        size = self.param_shape if self.size == 'explicit' else None
        initial_avg_mean = None if self.mean is None else self.mean.copy()
        initial_avg_var = None if self.var is None else self.var.copy()

        link = links.BatchNormalization(
            size=size,
            axis=self.aggr_axes,
            eps=self.eps,
            dtype=self.dtype,
            initial_gamma=initial_gamma,
            initial_beta=initial_beta,
            initial_avg_mean=initial_avg_mean,
            initial_avg_var=initial_avg_var)
        return link

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, link, inputs, device):
        x, = inputs

        # The inputs might be of different dtype than what the link was
        # initialized with. In that case, persistent values must be manually
        # cast. This is needed when forward is called in order to compute
        # numerical gradients.
        if ((self.dtype == chainer.mixed16 and x.dtype != numpy.float16)
            or (self.dtype != chainer.mixed16 and link.avg_mean is not None
                and x.dtype != link.avg_mean.dtype)):
            link.avg_mean = link.avg_mean.astype(x.dtype)
            link.avg_var = link.avg_var.astype(x.dtype)

        with chainer.using_config('train', not self.test):
            y = link(x, finetune=self.finetune)

        return y,

    def forward_expected(self, link, inputs):
        gamma = link.gamma.array
        beta = link.beta.array
        x, = inputs

        if self.test:
            mean = self.mean[self.expander]
            var = self.var[self.expander]
            std = numpy.sqrt(var)
        else:
            mean = x.mean(
                axis=self.aggr_axes, dtype=self.highprec_dtype, keepdims=True)
            var = x.var(
                axis=self.aggr_axes, dtype=self.highprec_dtype, keepdims=True)
            std = numpy.sqrt(var + self.eps)
        y = gamma[self.expander] * (x - mean) / std + beta[self.expander]

        if self.dtype == chainer.mixed16:
            y = y.astype(x.dtype)

        return y,

    def check_forward_outputs(self, outputs, expected_outputs):
        super(BatchNormalizationTestBase, self).check_forward_outputs(
            outputs, expected_outputs)
        y, = outputs
        assert y.dtype == chainer.get_dtype(self.dtype)


@_inject_backend_tests
@_parameterize
class BatchNormalizationTest(BatchNormalizationTestBase, testing.LinkTestCase):

    pass


# TODO(hvy): Safely remove this test class when BackendConfig no longer
# modifies the current device since those cases should be covered by the tests
# above.
@attr.multi_gpu(2)
@testing.inject_backend_tests(
    None,
    testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0],
    }))
@_parameterize
class BatchNormalizationMultiGpuTest(
        BatchNormalizationTestBase, testing.LinkTestCase):

    skip_backward_test = True

    # TODO(hvy): Remove this relaxation. It is currently needed as the
    # inter-device copy in CuPy with non-contiguous arrays are broken.
    contiguous = 'C'

    def forward(self, link, inputs, device):
        x, = inputs

        device_1 = backend.GpuDevice.from_device_id(1)
        link.to_device(device_1)
        x.to_device(device_1)

        device_0 = backend.GpuDevice.from_device_id(0)
        with chainer.using_device(device_0):
            with chainer.using_config('train', not self.test):
                y = link(x, finetune=self.finetune)
        return y,


@testing.parameterize(*(testing.product_dict(
    testing.product({
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
    }))))
@_inject_backend_tests
class BatchNormalizationInitializersTest(
        BatchNormalizationTestBase, testing.LinkInitializersTestCase):

    test = False
    size = 'skip'
    ndim = 1
    input_shape = (5, 4)
    axis = 0

    def get_initializers(self):
        initial_gamma = [
            initializers.Constant(2), 2, testing.InitializerArgument(None, 1)]
        initial_beta = [
            initializers.Constant(2), 2, testing.InitializerArgument(None, 0)]
        return initial_gamma, initial_beta


@testing.parameterize(
    {'nx': 10, 'ny': 10, 'eps': 2e-5},
    {'nx': 10, 'ny': 10, 'eps': 1e-1},
    # TODO(Kenta Oono)
    # Pass the case below (this test does not pass when nx != ny).
    # {'nx': 10, 'ny': 15}
)
class TestPopulationStatistics(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.size = 3
        self.link = links.BatchNormalization(self.size, self.decay, self.eps)
        self.x = numpy.random.uniform(
            -1, 1, (self.nx, self.size)).astype(numpy.float32)
        self.y = numpy.random.uniform(
            -1, 1, (self.ny, self.size)).astype(numpy.float32)

    def check_statistics(self, x, y):
        x = chainer.Variable(x)
        self.link(x, finetune=True)
        mean = self.x.mean(axis=0)
        testing.assert_allclose(mean, self.link.avg_mean)
        unbiased_var = self.x.var(axis=0) * self.nx / (self.nx - 1)
        testing.assert_allclose(unbiased_var, self.link.avg_var)

        y = chainer.Variable(y)
        with chainer.using_config('train', False):
            self.link(y, finetune=True)
        testing.assert_allclose(mean, self.link.avg_mean)
        testing.assert_allclose(unbiased_var, self.link.avg_var)

    def test_statistics_cpu(self):
        self.check_statistics(self.x, self.y)

    @attr.gpu
    def test_statistics_gpu(self):
        self.link.to_gpu()
        self.check_statistics(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

    @attr.cudnn
    def test_statistics_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_statistics_gpu()

    def check_statistics2(self, x, y):
        x = chainer.Variable(x)
        y = chainer.Variable(y)
        self.link(x, finetune=True)
        self.link(y, finetune=True)
        mean = (self.x.sum(axis=0) + self.y.sum(axis=0)) / (self.nx + self.ny)
        var = (self.x.var(axis=0) * self.nx +
               self.y.var(axis=0) * self.ny) / (self.nx + self.ny)

        # TODO(Kenta Oono)
        # Fix the estimate of the unbiased variance.
        # Unbiased variance should be (nx + ny) / (nx + ny - 1) times of
        # the variance.
        # But the multiplier is ny / (ny - 1) in current implementation
        # these two values are different when nx is not equal to ny.
        unbiased_var = var * self.ny / (self.ny - 1)
        testing.assert_allclose(mean, self.link.avg_mean)
        testing.assert_allclose(unbiased_var, self.link.avg_var)

    def test_statistics2_cpu(self):
        self.check_statistics2(self.x, self.y)

    @attr.gpu
    def test_statistics2_gpu(self):
        self.link.to_gpu()
        self.check_statistics2(
            cuda.to_gpu(self.x),
            cuda.to_gpu(self.y))

    @attr.cudnn
    def test_statistics2_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_statistics2_gpu()


# TODO(hvy): Rewrite this test using testing.LinkTestCase.
@testing.parameterize(*testing.product({
    'test': [True, False],
    'ndim': [0, 1, 2, 3],
}))
class BatchNormalizationTestWithoutGammaAndBeta(unittest.TestCase):

    def setUp(self):
        self.link = links.BatchNormalization(
            3, use_gamma=False, use_beta=False)
        if self.test:
            mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
            self.link.avg_mean[...] = mean
            var = numpy.random.uniform(0.5, 1, (3,)).astype(numpy.float32)
            self.link.avg_var[...] = var
        self.link.cleargrads()

        shape = (7, 3) + (2,) * self.ndim
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)

        expander = (None, Ellipsis) + (None,) * self.ndim
        gamma = numpy.ones((3,), dtype=numpy.float32)[expander]
        beta = numpy.zeros((3,), dtype=numpy.float32)[expander]
        if self.test:
            mean = self.link.avg_mean
            var = self.link.avg_var
        else:
            aggr_axes = (0,) + tuple(six.moves.range(2, self.ndim + 2))
            mean = self.x.mean(axis=aggr_axes)
            var = self.x.var(axis=aggr_axes)
        self.y_expected = _batch_normalization(
            expander, gamma, beta, self.x, mean, var, self.link.eps, self.test)

    def test_no_gamma_and_beta(self):
        assert self.link.gamma is None
        assert self.link.beta is None

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        with chainer.using_config('train', not self.test):
            y = self.link(x)
        testing.assert_allclose(self.y_expected, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        self.check_forward(x)

    @attr.multi_gpu(2)
    def test_forward_gpu_multi(self):
        with cuda.get_device_from_id(0):
            self.link.to_gpu()
            x = cuda.to_gpu(self.x)
        with cuda.get_device_from_id(1):
            self.check_forward(x)

    @attr.cudnn
    def test_forward_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_forward_gpu()

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(self.link, x_data, y_grad,
                                      eps=1e-2, rtol=1e-3, atol=1e-4)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        x = cuda.to_gpu(self.x)
        gy = cuda.to_gpu(self.gy)
        self.check_backward(x, gy)

    @attr.cudnn
    def test_backward_gpu_without_cudnn(self):
        with chainer.using_config('use_cudnn', 'never'):
            self.test_backward_gpu()


def _generate_uniform(low, high, shape, dtype=numpy.float32):
    return numpy.random.uniform(low, high, shape).astype(dtype)


@testing.parameterize(*testing.product({
    'size': [3, (2, 3)],
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.initial_gamma = _generate_uniform(-1, 1, self.size)
        self.initial_beta = _generate_uniform(-1, 1, self.size)
        self.initial_avg_mean = _generate_uniform(-1, 1, self.size)
        self.initial_avg_var = _generate_uniform(-1, 1, self.size)
        self.link = links.BatchNormalization(
            self.size, self.decay,
            initial_gamma=self.initial_gamma,
            initial_beta=self.initial_beta,
            initial_avg_mean=self.initial_avg_mean,
            initial_avg_var=self.initial_avg_var,
        )

    @condition.retry(3)
    def test_initialize_cpu(self):
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)
        testing.assert_allclose(self.initial_avg_mean, self.link.avg_mean)
        testing.assert_allclose(self.initial_avg_var, self.link.avg_var)

    @attr.gpu
    @condition.retry(3)
    def test_initialize_gpu(self):
        self.link.to_gpu()
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)
        testing.assert_allclose(self.initial_avg_mean, self.link.avg_mean)
        testing.assert_allclose(self.initial_avg_var, self.link.avg_var)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float16, chainer.mixed16],
}))
class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.decay = 0.9
        self.size = 3
        with chainer.using_config('dtype', self.dtype):
            self.link = links.BatchNormalization(self.size, self.decay)
        dtype = numpy.float32 if self.dtype == chainer.mixed16 else self.dtype
        assert self.link.beta.dtype == dtype
        assert self.link.gamma.dtype == dtype
        assert self.link.avg_mean.dtype == dtype
        assert self.link.avg_var.dtype == dtype

        self.x = numpy.arange(6, dtype=self.dtype).reshape(2, 3)

    def check_initialize(self):
        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.array)
        testing.assert_allclose(numpy.zeros(self.size), self.link.beta.array)
        testing.assert_allclose(0, self.link.avg_mean)
        testing.assert_allclose(1, self.link.avg_var)
        y = self.link(self.x)
        assert y.dtype == self.x.dtype

    def test_initialize_cpu(self):
        self.check_initialize()

    @attr.gpu
    def test_initialize_gpu(self):
        self.link.to_gpu()
        self.x = cuda.to_gpu(self.x)
        self.check_initialize()


@testing.parameterize(*testing.product({
    'shape': [(2, 4), (2, 5, 3, 4)],
}))
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.link = links.BatchNormalization(3)

    def test_invalid_shape_cpu(self):
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(numpy.zeros(self.shape, dtype='f')))

    @attr.gpu
    def test_invalid_shape_gpu(self):
        self.link.to_gpu()
        with self.assertRaises(type_check.InvalidType):
            self.link(chainer.Variable(cuda.cupy.zeros(self.shape, dtype='f')))


class TestInvalidInitialize(unittest.TestCase):

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            self.link = links.BatchNormalization({})


class TestInvalidArgument(unittest.TestCase):

    def setUp(self):
        self.link = links.BatchNormalization(1)
        self.x = numpy.random.uniform(-1, 1, (3,)).astype('f')

    def test_test_argument(self):
        with self.assertRaises(ValueError):
            self.link(self.x, test=True)

    def test_positional_argument(self):
        # positional argument is prohibited from v2
        with self.assertRaises(TypeError):
            self.link(self.x, True)

    def test_redundant_argument(self):
        with self.assertRaises(TypeError):
            self.link(self.x, unknown_argument=1)


@testing.parameterize(
    {'shape': (5, 4, 3, 2), 'axis': (0, 2, 3)},
    {'shape': (5, 4), 'axis': 0},
    {'shape': (5, 4, 3), 'axis': (0, 1)},
)
class TestChannalSizeInference(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.randn(*self.shape).astype('f')

        axis = self.axis
        if isinstance(axis, int):
            axis = (axis,)
        self.expected_size = tuple(
            n
            for i, n in enumerate(self.shape)
            if i not in axis
        )

    def test_no_inference(self):
        bn = links.BatchNormalization(self.expected_size)
        assert bn.avg_mean is not None
        assert bn.avg_var is not None

    def test_inference_cpu(self):
        bn = links.BatchNormalization(axis=self.axis)
        bn(self.x)
        assert bn.beta.shape == self.expected_size
        assert bn.gamma.shape == self.expected_size
        assert bn.avg_mean.shape == self.expected_size
        assert bn.avg_var.shape == self.expected_size

    @attr.gpu
    def test_inference_gpu(self):
        bn = links.BatchNormalization(axis=self.axis)
        bn.to_gpu()
        bn(cuda.to_gpu(self.x))
        assert isinstance(bn.beta.data, cuda.cupy.ndarray)
        assert isinstance(bn.gamma.data, cuda.cupy.ndarray)
        assert isinstance(bn.avg_mean, cuda.cupy.ndarray)
        assert isinstance(bn.avg_var, cuda.cupy.ndarray)
        assert bn.beta.shape == self.expected_size
        assert bn.gamma.shape == self.expected_size
        assert bn.avg_mean.shape == self.expected_size
        assert bn.avg_var.shape == self.expected_size

    def test_no_gamma(self):
        bn = links.BatchNormalization(axis=self.axis, use_gamma=False)
        assert bn.gamma is None
        bn(self.x)
        assert bn.gamma is None

    def test_no_beta(self):
        bn = links.BatchNormalization(axis=self.axis, use_beta=False)
        assert bn.beta is None
        bn(self.x)
        assert bn.beta is None


class TestFailChannalSizeInference(unittest.TestCase):

    def test_fail_inference(self):
        with self.assertRaises(RuntimeError):
            links.BatchNormalization()


@attr.multi_gpu(2)
class TestLazyInitializationWithNonZeroCurrentCudaDevice(unittest.TestCase):

    def test_lazy_initialization_with_non_zero_current_cuda_device(self):
        # Create a lazily initialized BatchNormalization link.
        bn = links.BatchNormalization(axis=(0, 2, 3))
        assert bn.xp is numpy

        device = backend.GpuDevice.from_device_id(1)
        bn.to_device(device)
        assert bn.xp is cuda.cupy
        assert bn.device == device
        assert bn.beta.device == device
        assert bn.gamma.device == device
        assert bn.avg_mean is None
        assert bn.avg_var is None

        x = numpy.random.randn(5, 4, 3, 2).astype(numpy.float32)
        x = device.send(x)

        # All parameters and persistent values should correctly be initialized
        # on device 1, and not device 0, meaning forward pass should not raise
        # any errors.
        bn(x)
        assert bn.xp is cuda.cupy
        assert bn.device == device
        assert bn.beta.device == device
        assert bn.gamma.device == device
        assert bn.avg_mean is not None
        assert bn.avg_var is not None
        assert backend.GpuDevice.from_array(bn.avg_mean) == device
        assert backend.GpuDevice.from_array(bn.avg_var) == device


testing.run_module(__name__, __file__)

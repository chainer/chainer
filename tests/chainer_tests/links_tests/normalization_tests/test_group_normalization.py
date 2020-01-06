import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.inject_backend_tests(
    None,
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
)
@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 3), (5, 4, 7), (3, 20)],
    'groups': [1, 2, 4],
    'dtype': [numpy.float16, numpy.float32, numpy.float64,
              chainer.mixed16],
})))
class GroupNormalizationTest(testing.LinkTestCase):

    param_names = ('gamma', 'beta')

    def setUp(self):
        self.x, = self.generate_inputs()
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

        if self.dtype in (numpy.float16, chainer.mixed16):
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-1}
            self.check_backward_options = {'atol': 5e-1, 'rtol': 1e-1}
        else:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def create_link(self, initializers):
        initial_gamma, initial_beta = initializers
        with chainer.using_config('dtype', self.dtype):
            link = links.GroupNormalization(
                self.groups,
                initial_gamma=initial_gamma,
                initial_beta=initial_beta,
            )
        return link

    def generate_params(self):
        highprec_dtype = chainer.get_dtype(
            self.dtype, map_mixed16=numpy.float32)
        initial_gamma = numpy.random.uniform(
            -1, 1, (self.shape[1],)).astype(highprec_dtype)
        initial_beta = numpy.random.uniform(
            -1, 1, (self.shape[1],)).astype(highprec_dtype)
        return initial_gamma, initial_beta

    def generate_inputs(self):
        shape = self.shape

        # sample x such that x.std >= min_std
        min_std = 0.02
        retry = 0
        while True:
            x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
            x_groups = x.reshape(shape[0], self.groups, -1)
            if x_groups.std(axis=2).min() >= min_std:
                break
            retry += 1
            assert retry <= 20, 'Too many retries to generate inputs'

        return x,

    def forward_expected(self, link, inputs):
        gamma = link.gamma.array
        beta = link.beta.array
        x, = inputs
        shape = self.shape
        param_reshape = tuple([
            s if i == 1 else 1 for i, s in enumerate(shape)])
        x = x.astype(chainer.get_dtype(
            self.dtype, map_mixed16=numpy.float32))
        x = x.reshape(shape[0] * self.groups, -1)
        x -= x.mean(axis=1, keepdims=True)
        x /= numpy.sqrt(link.eps + numpy.square(x).mean(axis=1, keepdims=True))
        x = x.reshape(shape)
        x = gamma.reshape(param_reshape) * x + beta.reshape(param_reshape)
        if self.dtype == chainer.mixed16:
            x = x.astype(numpy.float16)
        return x,


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64,
              chainer.mixed16],
}))
class TestInitialize(unittest.TestCase):

    def setUp(self):
        self.lowprec_dtype = chainer.get_dtype(self.dtype)
        highprec_dtype = chainer.get_dtype(
            self.dtype, map_mixed16=numpy.float32)

        self.initial_gamma = numpy.random.uniform(-1, 1, self.size)
        self.initial_gamma = self.initial_gamma.astype(highprec_dtype)
        self.initial_beta = numpy.random.uniform(-1, 1, self.size)
        self.initial_beta = self.initial_beta.astype(highprec_dtype)
        self.link = links.GroupNormalization(self.groups,
                                             initial_gamma=self.initial_gamma,
                                             initial_beta=self.initial_beta)
        self.shape = (1, self.size, 1)

    def test_initialize_cpu(self):
        with chainer.using_config('dtype', self.dtype):
            self.link(numpy.zeros(self.shape, dtype=self.lowprec_dtype))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)

    @attr.gpu
    def test_initialize_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        with chainer.using_config('dtype', self.dtype):
            self.link(cuda.cupy.zeros(self.shape, dtype=self.lowprec_dtype))
        testing.assert_allclose(self.initial_gamma, self.link.gamma.data)
        testing.assert_allclose(self.initial_beta, self.link.beta.data)


@testing.parameterize(*testing.product({
    'size': [3, 30],
    'groups': [1, 3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64,
              chainer.mixed16],
}))
class TestDefaultInitializer(unittest.TestCase):

    def setUp(self):
        self.lowprec_dtype = chainer.get_dtype(self.dtype)
        self.highprec_dtype = chainer.get_dtype(
            self.dtype, map_mixed16=numpy.float32)

        self.size = 3
        with chainer.using_config('dtype', self.dtype):
            self.link = links.GroupNormalization(self.groups)
        self.shape = (1, self.size, 1)

    def test_initialize_cpu(self):
        self.link(numpy.zeros(self.shape, dtype=self.lowprec_dtype))

        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        self.assertEqual(self.link.gamma.dtype, self.highprec_dtype)

        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)
        self.assertEqual(self.link.beta.dtype, self.highprec_dtype)

    @attr.gpu
    def test_initialize_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        self.link(cuda.cupy.zeros(self.shape, dtype=self.lowprec_dtype))

        testing.assert_allclose(numpy.ones(self.size), self.link.gamma.data)
        self.assertEqual(self.link.gamma.dtype, self.highprec_dtype)

        testing.assert_allclose(
            numpy.zeros(self.size), self.link.beta.data)
        self.assertEqual(self.link.beta.dtype, self.highprec_dtype)


@testing.parameterize(*testing.product({
    'shape': [(2,), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestInvalidInput(unittest.TestCase):

    def setUp(self):
        self.link = links.GroupNormalization(groups=3)

    def test_invalid_shape_cpu(self):
        with self.assertRaises(ValueError):
            self.link(chainer.Variable(
                numpy.zeros(self.shape, dtype=self.dtype)))

    @attr.gpu
    def test_invalid_shape_gpu(self):
        with testing.assert_warns(DeprecationWarning):
            self.link.to_gpu()
        with self.assertRaises(ValueError):
            self.link(
                chainer.Variable(
                    cuda.cupy.zeros(self.shape, dtype=self.dtype)))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestInvalidInitialize(unittest.TestCase):

    def setUp(self):
        shape = (2, 5, 2)
        self.x = chainer.Variable(numpy.zeros(shape, dtype=self.dtype))

    def test_invalid_groups(self):
        self.link = links.GroupNormalization(groups=3)
        with self.assertRaises(ValueError):
            self.link(self.x)

    def test_invalid_type_groups(self):
        self.link = links.GroupNormalization(groups=3.5)
        with self.assertRaises(TypeError):
            self.link(self.x)


testing.run_module(__name__, __file__)

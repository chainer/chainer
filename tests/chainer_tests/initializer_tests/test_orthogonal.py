import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (), 'dim_in': 1, 'dim_out': 1},
        {'shape': (1,), 'dim_in': 1, 'dim_out': 1},
        {'shape': (3, 4), 'dim_in': 4, 'dim_out': 3},
        {'shape': (3, 4, 5), 'dim_in': 20, 'dim_out': 3}
    ],
    [
        {'scale': 2., 'dtype': numpy.float16}
    ] + testing.product({
        'scale': [None, 7.3],
        'dtype': [numpy.float32, numpy.float64],
    })
))
@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_ideep': 'always'},
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
    ]
)
class OrthogonalBase(unittest.TestCase):

    target = initializers.Orthogonal

    def setUp(self):
        kwargs = {}
        if self.scale is not None:
            kwargs['scale'] = self.scale
        self.target_kwargs = kwargs

        self.check_options = {}
        if self.dtype == numpy.float16:
            self.check_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_initializer(self, w):
        initializer = self.target(**self.target_kwargs)
        initializer(w)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_initializer(self, backend_config):
        w = numpy.empty(self.shape, dtype=self.dtype)
        w = backend_config.get_array(w)
        with chainer.using_device(backend_config.device):
            self.check_initializer(w)

    def check_shaped_initializer(self, backend_config):
        initializer = self.target(dtype=self.dtype, **self.target_kwargs)
        xp = backend_config.xp
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(backend.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_shaped_initializer(self, backend_config):
        with chainer.using_device(backend_config.device):
            self.check_shaped_initializer(backend_config)

    def check_orthogonality(self, w):
        initializer = self.target(**self.target_kwargs)
        initializer(w)
        n = self.dim_out
        w = w.astype(numpy.float64).reshape(n, -1)
        dots = w.dot(w.T)
        expected_scale = self.scale or 1.1
        testing.assert_allclose(
            dots, numpy.identity(n) * expected_scale**2,
            **self.check_options)

    def test_orthogonality(self, backend_config):
        w = numpy.empty(self.shape, dtype=self.dtype)
        w = backend_config.get_array(w)
        with chainer.using_device(backend_config.device):
            self.check_orthogonality(w)

    def check_initializer_statistics(self, backend_config, n):
        from scipy import stats
        xp = backend_config.xp
        ws = numpy.empty((n,) + self.shape, dtype=self.dtype)
        ws = backend_config.get_array(ws)
        for i in range(n):
            initializer = self.target(**self.target_kwargs)
            initializer(xp.squeeze(ws[i:i+1], axis=0))

        expected_scale = self.scale or 1.1
        sampless = cuda.to_cpu(ws.reshape(n, -1).T)
        alpha = 0.01 / len(sampless)

        ab = 0.5 * (self.dim_in - 1)

        for samples in sampless:
            if self.dim_in == 1:
                numpy.testing.assert_allclose(abs(samples), expected_scale)
                _, p = stats.chisquare((numpy.sign(samples) + 1) // 2)
            else:
                _, p = stats.kstest(
                    samples,
                    stats.beta(
                        ab, ab,
                        loc=-expected_scale,
                        scale=2*expected_scale
                    ).cdf
                )
            assert p >= alpha

    @testing.with_requires('scipy')
    @condition.retry(3)
    def test_initializer_statistics(self, backend_config):
        with chainer.using_device(backend_config.device):
            self.check_initializer_statistics(backend_config, 100)

    @attr.slow
    @testing.with_requires('scipy')
    @condition.repeat_with_success_at_least(5, 3)
    def test_initializer_statistics_slow(self, backend_config):
        with chainer.using_device(backend_config.device):
            self.check_initializer_statistics(backend_config, 10000)


class TestEmpty(unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(0, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal()

    def check_assert(self, w):
        with self.assertRaises(ValueError):
            self.initializer(w)

    def test_cpu(self):
        self.check_assert(self.w)

    @attr.gpu
    def test_gpu(self):
        self.check_assert(cuda.to_gpu(self.w))


@testing.parameterize(
    {'shape': (4, 3)},
    {'shape': (21, 4, 5)})
class TestOverComplete(unittest.TestCase):

    def setUp(self):
        self.w = numpy.empty(self.shape, dtype=numpy.float32)
        self.initializer = initializers.Orthogonal(scale=1.0)

    def check_invalid(self, w):
        with self.assertRaises(ValueError):
            self.initializer(w)

    def test_invalid_cpu(self):
        self.check_invalid(self.w)

    @attr.gpu
    def test_invalid_gpu(self):
        self.check_invalid(cuda.to_gpu(self.w))


testing.run_module(__name__, __file__)

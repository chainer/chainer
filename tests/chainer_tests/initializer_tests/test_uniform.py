import math
import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


default_scale = {
    initializers.Uniform: 0.05,
}

default_coeff = {
    initializers.LeCunUniform: math.sqrt(3),
    initializers.GlorotUniform: math.sqrt(3),
    initializers.HeUniform: math.sqrt(6),
}

default_fan = {
    initializers.LeCunUniform: 'fan_in',
    initializers.GlorotUniform: 'fan_avg',
    initializers.HeUniform: 'fan_in',
}


@testing.parameterize(*testing.product({
    'target,fan_option': [
        (initializers.Uniform, None),
        (initializers.LeCunUniform, None),
        (initializers.GlorotUniform, None),
        (initializers.HeUniform, None),
    ],
    'shape,fans': [
        ((2, 3), (3, 2)),
        ((2, 3, 4), (12, 8)),
    ],
    'scale': [None, 7.3],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'rng_class': [None, numpy.random.RandomState],
}))
@testing.backend.inject_backend_tests(
    None,
    [
        {},
        {'use_ideep': 'always'},
        {'use_cuda': True, 'cuda_device': 0},
        {'use_cuda': True, 'cuda_device': 1},
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ]
)
class TestUniform(unittest.TestCase):

    def setUp(self):
        kwargs = {}
        if self.scale is not None:
            kwargs['scale'] = self.scale
        if self.fan_option is not None:
            kwargs['fan_option'] = self.fan_option
        if self.rng_class is not None:
            kwargs['rng'] = self.rng_class()
        self.target_kwargs = kwargs

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

    def check_initializer_statistics(self, backend_config, n):
        from scipy import stats

        xp = backend_config.xp
        ws = numpy.empty((n,) + self.shape, dtype=self.dtype)
        ws = backend_config.get_array(ws)
        for i in range(n):
            initializer = self.target(**self.target_kwargs)
            initializer(xp.squeeze(ws[i:i+1], axis=0))

        fan = self.fan_option or default_fan.get(self.target)
        expected_max = self.scale or default_scale.get(self.target) or 1.
        expected_max *= default_coeff.get(self.target) or 1.
        if fan is not None:
            if fan == 'fan_in':
                expected_max *= math.sqrt(1. / self.fans[0])
            elif fan == 'fan_out':
                expected_max *= math.sqrt(1. / self.fans[1])
            elif fan == 'fan_avg':
                expected_max *= math.sqrt(2. / sum(self.fans))
            else:
                assert False

        sampless = cuda.to_cpu(ws.reshape(n, -1).T)
        alpha = 0.01 / len(sampless)
        for samples in sampless:
            _, p = stats.kstest(
                samples,
                stats.uniform(-expected_max, 2*expected_max).cdf
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


testing.run_module(__name__, __file__)

import math
import unittest

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


default_scale = {
    initializers.Normal: 0.05,
}

default_coeff = {
    initializers.HeNormal: math.sqrt(2),
}

default_fan = {
    initializers.LeCunNormal: 'fan_in',
    initializers.GlorotNormal: 'fan_avg',
    initializers.HeNormal: 'fan_in',
}


@testing.parameterize(*testing.product_dict(
    [
        {'target': initializers.Normal, 'fan_option': None},
        {'target': initializers.LeCunNormal, 'fan_option': None},
        {'target': initializers.GlorotNormal, 'fan_option': None},
        {'target': initializers.HeNormal, 'fan_option': 'fan_in'},
        {'target': initializers.HeNormal, 'fan_option': 'fan_out'}
    ],
    [
        {'shape': (2, 3), 'fans': (3, 2)},
        {'shape': (2, 3, 4), 'fans': (12, 8)},
    ],
    testing.product({
        'scale': [None, 7.3],
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
    })
))
class NormalBase(unittest.TestCase):

    def setUp(self):
        kwargs = {}
        if self.scale is not None:
            kwargs['scale'] = self.scale
        if self.fan_option is not None:
            kwargs['fan_option'] = self.fan_option
        self.target_kwargs = kwargs

    def check_initializer(self, w):
        initializer = self.target(**self.target_kwargs)
        initializer(w)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_initializer_cpu(self):
        w = numpy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    @attr.gpu
    def test_initializer_gpu(self):
        w = cuda.cupy.empty(self.shape, dtype=self.dtype)
        self.check_initializer(w)

    def check_shaped_initializer(self, xp):
        initializer = self.target(dtype=self.dtype, **self.target_kwargs)
        w = initializers.generate_array(initializer, self.shape, xp)
        self.assertIs(backend.get_array_module(w), xp)
        self.assertTupleEqual(w.shape, self.shape)
        self.assertEqual(w.dtype, self.dtype)

    def test_shaped_initializer_cpu(self):
        self.check_shaped_initializer(numpy)

    @attr.gpu
    def test_shaped_initializer_gpu(self):
        self.check_shaped_initializer(cuda.cupy)

    def check_initializer_statistics(self, xp, n):
        from scipy import stats

        ws = xp.empty((n,) + self.shape, dtype=self.dtype)
        for i in range(n):
            initializer = self.target(**self.target_kwargs)
            initializer(xp.squeeze(ws[i:i+1], axis=0))

        fan = self.fan_option or default_fan.get(self.target)
        expected_std = self.scale or default_scale.get(self.target) or 1.
        expected_std *= default_coeff.get(self.target) or 1.
        if fan is not None:
            if fan == 'fan_in':
                expected_std *= math.sqrt(1. / self.fans[0])
            elif fan == 'fan_out':
                expected_std *= math.sqrt(1. / self.fans[1])
            elif fan == 'fan_avg':
                expected_std *= math.sqrt(2. / sum(self.fans))
            else:
                assert False

        sampless = cuda.to_cpu(ws.reshape(n, -1).T)
        alpha = 0.01 / len(sampless)
        for samples in sampless:
            _, p = stats.kstest(samples, stats.norm(0, expected_std).cdf)
            assert p >= alpha

    @testing.with_requires('scipy')
    @condition.retry(3)
    def test_initializer_statistics_cpu(self):
        self.check_initializer_statistics(numpy, 100)

    @attr.gpu
    @testing.with_requires('scipy')
    @condition.retry(3)
    def test_initializer_statistics_gpu(self):
        self.check_initializer_statistics(cuda.cupy, 100)

    @attr.slow
    @testing.with_requires('scipy')
    @condition.repeat_with_success_at_least(5, 3)
    def test_initializer_statistics_slow_cpu(self):
        self.check_initializer_statistics(numpy, 10000)

    @attr.slow
    @attr.gpu
    @testing.with_requires('scipy')
    @condition.repeat_with_success_at_least(5, 3)
    def test_initializer_statistics_slow_gpu(self):
        self.check_initializer_statistics(cuda.cupy, 10000)


testing.run_module(__name__, __file__)

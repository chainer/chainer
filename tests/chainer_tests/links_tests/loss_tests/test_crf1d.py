import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import initializers
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64]
}))
class TestCRF1d(unittest.TestCase):

    n_labels = 3

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()

        self.xs = [numpy.array([[1, 0, 0], [1, 0, 0]], self.dtype),
                   numpy.array([[0, 1, 0], [0, 1, 0]], self.dtype),
                   numpy.array([[0, 1, 0]], self.dtype)]

        self.ts = [numpy.array([0, 0], numpy.int32),
                   numpy.array([1, 1], numpy.int32),
                   numpy.array([2], numpy.int32)]

        self.link = links.CRF1d(n_label=self.n_labels)
        self.cost_shape = (self.n_labels, self.n_labels)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3}
        else:
            self.check_forward_options = {'atol': 1e-4}

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def check_forward(self, x_data, t_data):
        self.link(x_data, t_data)

    def test_forward_cpu(self):
        print(self.xs, self.ts)
        self.check_forward(self.xs, self.ts)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.xs), cuda.to_gpu(self.ts))

    def test_zeroinit(self):
        link = links.CRF1d(n_label=self.n_labels)
        x = link.cost.data
        t = link.xp.zeros(self.cost_shape, dtype=link.xp.float32)
        self.assertTrue(link.xp.array_equal(x, t))

    def test_xavierinit(self):
        link = links.CRF1d(n_label=self.n_labels,
                           initial_cost=initializers.GlorotUniform())
        x = link.cost.data
        t = link.xp.zeros(self.cost_shape, dtype=link.xp.float32)
        self.assertFalse(link.xp.array_equal(x, t))


testing.run_module(__name__, __file__)

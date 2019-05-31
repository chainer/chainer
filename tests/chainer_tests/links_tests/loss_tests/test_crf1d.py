import unittest

import itertools
import numpy
from six import moves

import chainer
from chainer.backends import cuda
from chainer import initializers
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'initial_cost': ['random', None],
    'transpose': [True, False],
}))
class TestCRF1d(unittest.TestCase):
    def _calc_score(self, batch, ys):
        cost = self.link.cost.array
        return sum(x[batch, y] for x, y in zip(self.xs, ys)) + \
            sum(cost[y1, y2] for y1, y2 in zip(ys[:-1], ys[1:]))

    def _crf1d(self, cost_data, xs_data, ys_data):
        z = numpy.zeros((self.batches[0],), numpy.float32)
        for b, length in enumerate(self.lengths):
            for ys in itertools.product(range(self.n_label), repeat=length):
                z[b] += numpy.exp(chainer.cuda.to_cpu(self._calc_score(b, ys)))

        score = numpy.zeros((self.batches[0],), numpy.float32)
        for b, length in enumerate(self.lengths):
            ys = [self.ys[i][b] for i in range(length)]
            score[b] = self._calc_score(b, ys)

        loss = -(score - numpy.log(z))
        return numpy.sum(loss) / self.batches[0]

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()
        self.n_label = 3

        self.lengths = [3, 3]
        self.batches = [2, 2, 2]

        self.xs = [numpy.random.uniform(-1, 1, (b, 3)).astype(self.dtype)
                   for b in self.batches]
        self.ys = [numpy.random.randint(
            0, self.n_label, (b,)).astype(numpy.int32)
            for b in self.batches]

        self.link = links.CRF1d(n_label=self.n_label)
        self.cost_shape = (self.n_label, self.n_label)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3}
        else:
            self.check_forward_options = {'atol': 1e-4}

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def check_forward(self, x_data, t_data):
        if self.transpose:
            # Make transposed arrays manually
            xs = [self.link.xp.empty((l, 3), dtype=self.dtype)
                  for l in self.lengths]
            ts = [self.link.xp.empty((l,), dtype=numpy.int32)
                  for l in self.lengths]
            for i, batch in enumerate(self.batches):
                for j in moves.range(batch):
                    xs[j][i] = x_data[i][j]
                    ts[j][i] = t_data[i][j]
        else:
            xs = x_data
            ts = t_data

        x = self.link(xs, ts, transpose=self.transpose)
        t = self._crf1d(self.link.cost.array, x_data, t_data)
        testing.assert_allclose(x.array, t,
                                **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.xs, self.ys)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.xs), cuda.to_gpu(self.ys))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'initializer': ['random', None]
}))
class TestInitialization(unittest.TestCase):

    def setUp(self):
        self.n_label = 3
        self.initial_cost = numpy.empty((self.n_label, self.n_label),
                                        dtype=self.dtype)

        if self.initializer is None:
            initializer = initializers.constant.Zero()

        elif self.initializer == 'random':
            initializer = initializers.GlorotUniform()

        initializer(self.initial_cost)
        with chainer.using_config('dtype', self.dtype):
            self.link = links.CRF1d(self.n_label,
                                    initial_cost=self.initial_cost)

    def check_param(self):
        link = self.link
        dtype = self.dtype
        assert link.cost.dtype == dtype
        testing.assert_allclose(link.cost.array,
                                self.initial_cost,
                                atol=0, rtol=0)

    def test_param_cpu(self):
        self.check_param()

    @attr.gpu
    def test_param_gpu(self):
        self.link.to_gpu()
        self.check_param()


testing.run_module(__name__, __file__)

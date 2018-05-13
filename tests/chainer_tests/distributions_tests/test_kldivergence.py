import unittest

import chainer
from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr
import numpy


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
}))
@testing.fix_random()
class TestKLDivergence(unittest.TestCase):

    def check_kl(self, dist1, dist2):
        kl = distributions.kl_divergence(dist1, dist2).data

        sample = dist1.sample(100000)
        mc_kl = dist1.log_prob(sample).data - dist2.log_prob(sample).data
        mc_kl = numpy.mean(mc_kl, axis=0)

        if isinstance(kl, cuda.ndarray):
            kl = kl.get()
        if isinstance(mc_kl, cuda.ndarray):
            mc_kl = mc_kl.get()

        testing.assert_allclose(kl, mc_kl, atol=1e-2, rtol=1e-2)

    def encode_params(self, params, is_gpu=False):
        if is_gpu:
            params = {k: cuda.to_gpu(v) for k, v in params.items()}

        if self.is_variable:
            params = {k: chainer.Variable(v) for k, v in params.items()}

        return params

    def make_normal_params(self, is_gpu=False):
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        return self.encode_params({"loc": loc, "scale": scale}, is_gpu)

    def test_normal_cpu(self):
        dist1 = distributions.Normal(**self.make_normal_params())
        dist2 = distributions.Normal(**self.make_normal_params())
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_normal_gpu(self):
        dist1 = distributions.Normal(**self.make_normal_params(True))
        dist2 = distributions.Normal(**self.make_normal_params(True))
        self.check_kl(dist1, dist2)

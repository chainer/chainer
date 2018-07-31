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
        kl = chainer.kl_divergence(dist1, dist2).data
        if isinstance(kl, cuda.ndarray):
            kl = kl.get()

        sample = dist1.sample(300000)
        mc_kl = dist1.log_prob(sample).data - dist2.log_prob(sample).data
        if isinstance(mc_kl, cuda.ndarray):
            mc_kl = mc_kl.get()
        mc_kl = numpy.nanmean(mc_kl, axis=0)

        print(kl, mc_kl)
        testing.assert_allclose(kl, mc_kl, atol=1e-2, rtol=1e-2)

    def encode_params(self, params, is_gpu=False):
        if is_gpu:
            params = {k: cuda.to_gpu(v) for k, v in params.items()}

        if self.is_variable:
            params = {k: chainer.Variable(v) for k, v in params.items()}

        return params

    def make_bernoulli_dist(self, is_gpu=False):
        p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)
        params = self.encode_params({"p": p}, is_gpu)
        return distributions.Bernoulli(**params)

    def make_laplace_dist(self, is_gpu=False):
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        params = self.encode_params({"loc": loc, "scale": scale}, is_gpu)
        return distributions.Laplace(**params)

    def make_normal_dist(self, is_gpu=False, use_log_scale=False):
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        if use_log_scale:
            log_scale = numpy.random.uniform(
                -1, 1, self.shape).astype(numpy.float32)
            params = self.encode_params(
                {"loc": loc, "log_scale": log_scale}, is_gpu)
        else:
            scale = numpy.exp(
                numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
            params = self.encode_params({"loc": loc, "scale": scale}, is_gpu)
        return distributions.Normal(**params)

    def make_multivariatenormal_dist(self, is_gpu=False):
        loc = numpy.random.uniform(
            -1, 1, self.shape + (3,)).astype(numpy.float32)
        cov = numpy.random.normal(size=(numpy.prod(self.shape),) + (3, 3))
        cov = [cov_.dot(cov_.T) for cov_ in cov]
        cov = numpy.vstack(cov).reshape(self.shape + (3, 3))
        scale_tril = numpy.linalg.cholesky(cov).astype(numpy.float32)
        params = self.encode_params(
            {"loc": loc, "scale_tril": scale_tril}, is_gpu)
        return distributions.MultivariateNormal(**params)

    def test_bernoulli_bernoulli_cpu(self):
        dist1 = self.make_bernoulli_dist()
        dist2 = self.make_bernoulli_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_bernoulli_bernoulli_gpu(self):
        dist1 = self.make_bernoulli_dist(True)
        dist2 = self.make_bernoulli_dist(True)
        self.check_kl(dist1, dist2)

    def test_laplace_laplace_cpu(self):
        dist1 = self.make_laplace_dist()
        dist2 = self.make_laplace_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_laplace_laplace_gpu(self):
        dist1 = self.make_laplace_dist(True)
        dist2 = self.make_laplace_dist(True)
        self.check_kl(dist1, dist2)

    def test_normal_normal_cpu(self):
        for use_log_scale1 in [True, False]:
            for use_log_scale2 in [True, False]:
                dist1 = self.make_normal_dist(use_log_scale=use_log_scale1)
                dist2 = self.make_normal_dist(use_log_scale=use_log_scale2)
                self.check_kl(dist1, dist2)

    @attr.gpu
    def test_normal_normal_gpu(self):
        for use_log_scale1 in [True, False]:
            for use_log_scale2 in [True, False]:
                dist1 = self.make_normal_dist(
                    True, use_log_scale=use_log_scale1)
                dist2 = self.make_normal_dist(
                    True, use_log_scale=use_log_scale2)
                self.check_kl(dist1, dist2)

    @testing.with_requires('scipy')
    def test_multivariatenormal_multivariatenormal_cpu(self):
        dist1 = self.make_multivariatenormal_dist()
        dist2 = self.make_multivariatenormal_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_multivariatenormal_multivariatenormal_gpu(self):
        dist1 = self.make_multivariatenormal_dist(True)
        dist2 = self.make_multivariatenormal_dist(True)
        self.check_kl(dist1, dist2)


testing.run_module(__name__, __file__)

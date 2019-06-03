import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import distributions
from chainer import testing
from chainer.testing import attr


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

        testing.assert_allclose(kl, mc_kl, atol=1e-2, rtol=1e-2)

    def encode_params(self, params, is_gpu=False):
        if is_gpu:
            params = {k: cuda.to_gpu(v) for k, v in params.items()}

        if self.is_variable:
            params = {k: chainer.Variable(v) for k, v in params.items()}

        return params

    def make_bernoulli_dist(self, is_gpu=False):
        p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)
        params = self.encode_params({'p': p}, is_gpu)
        return distributions.Bernoulli(**params)

    def make_beta_dist(self, is_gpu=False):
        a = numpy.random.uniform(0.5, 10, self.shape).astype(numpy.float32)
        b = numpy.random.uniform(0.5, 10, self.shape).astype(numpy.float32)
        params = self.encode_params({'a': a, 'b': b}, is_gpu)
        return distributions.Beta(**params)

    def make_categorical_dist(self, is_gpu=False):
        p = numpy.random.normal(size=self.shape+(3,)).astype(numpy.float32)
        p = numpy.exp(p)
        p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
        params = self.encode_params({'p': p}, is_gpu)
        return distributions.Categorical(**params)

    def make_dirichlet_dist(self, is_gpu=False):
        alpha = numpy.random.uniform(
            0.5, 10, self.shape + (3,)).astype(numpy.float32)
        params = self.encode_params({'alpha': alpha}, is_gpu)
        return distributions.Dirichlet(**params)

    def make_exponential_dist(self, is_gpu=False):
        lam = numpy.exp(
            numpy.random.uniform(0, 0.5, self.shape)).astype(numpy.float32)
        params = self.encode_params({'lam': lam}, is_gpu)
        return distributions.Exponential(**params)

    def make_gamma_dist(self, is_gpu=False):
        k = numpy.random.uniform(1, 5, self.shape).astype(numpy.float32)
        theta = numpy.random.uniform(0, 2, self.shape).astype(numpy.float32)
        params = self.encode_params({'k': k, 'theta': theta}, is_gpu)
        return distributions.Gamma(**params)

    def make_geometric_dist(self, is_gpu=False):
        p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)
        params = self.encode_params({'p': p}, is_gpu)
        return distributions.Geometric(**params)

    def make_gumbel_dist(self, is_gpu=False):
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(0, 1, self.shape)).astype(numpy.float32)
        params = self.encode_params({'loc': loc, 'scale': scale}, is_gpu)
        return distributions.Gumbel(**params)

    def make_laplace_dist(self, is_gpu=False):
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        scale = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        params = self.encode_params({'loc': loc, 'scale': scale}, is_gpu)
        return distributions.Laplace(**params)

    def make_log_normal_dist(self, is_gpu=False):
        mu = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        sigma = numpy.exp(
            numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
        params = self.encode_params({'mu': mu, 'sigma': sigma}, is_gpu)
        return distributions.LogNormal(**params)

    def make_normal_dist(self, is_gpu=False, use_log_scale=False):
        loc = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        if use_log_scale:
            log_scale = numpy.random.uniform(
                -1, 1, self.shape).astype(numpy.float32)
            params = self.encode_params(
                {'loc': loc, 'log_scale': log_scale}, is_gpu)
        else:
            scale = numpy.exp(
                numpy.random.uniform(-1, 1, self.shape)).astype(numpy.float32)
            params = self.encode_params({'loc': loc, 'scale': scale}, is_gpu)
        return distributions.Normal(**params)

    def make_multivariatenormal_dist(self, is_gpu=False):
        loc = numpy.random.uniform(
            -1, 1, self.shape + (3,)).astype(numpy.float32)
        cov = numpy.random.normal(size=(numpy.prod(self.shape),) + (3, 3))
        cov = [cov_.dot(cov_.T) for cov_ in cov]
        cov = numpy.vstack(cov).reshape(self.shape + (3, 3))
        scale_tril = numpy.linalg.cholesky(cov).astype(numpy.float32)
        params = self.encode_params(
            {'loc': loc, 'scale_tril': scale_tril}, is_gpu)
        return distributions.MultivariateNormal(**params)

    def make_one_hot_categorical_dist(self, is_gpu=False):
        p = numpy.random.normal(size=self.shape+(3,)).astype(numpy.float32)
        p = numpy.exp(p)
        p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
        params = self.encode_params({'p': p}, is_gpu)
        return distributions.OneHotCategorical(**params)

    def make_pareto_dist(self, is_gpu=False):
        scale = numpy.exp(numpy.random.uniform(
            0.5, 1, self.shape)).astype(numpy.float32)
        alpha = numpy.exp(numpy.random.uniform(
            1, 2, self.shape)).astype(numpy.float32)
        params = self.encode_params({'scale': scale, 'alpha': alpha}, is_gpu)
        return distributions.Pareto(**params)

    def make_poisson_dist(self, is_gpu=False):
        lam = numpy.random.uniform(5, 10, self.shape).astype(numpy.float32)
        params = self.encode_params({'lam': lam}, is_gpu)
        return distributions.Poisson(**params)

    def make_uniform_dist(self, is_gpu=False, low=None, high=None,
                          loc=None, scale=None, use_loc_scale=False):
        if use_loc_scale:
            if loc is None:
                loc = numpy.random.uniform(
                    -3, 0, self.shape).astype(numpy.float32)
            if scale is None:
                scale = numpy.random.uniform(
                    1, 5, self.shape).astype(numpy.float32)
            params = self.encode_params({'loc': loc, 'scale': scale}, is_gpu)
        else:
            if low is None:
                low = numpy.random.uniform(
                    -3, 0, self.shape).astype(numpy.float32)
            if high is None:
                high = numpy.random.uniform(
                    low + 1, low + 6, self.shape).astype(numpy.float32)
            params = self.encode_params({'low': low, 'high': high}, is_gpu)
        return distributions.Uniform(**params)

    def test_bernoulli_bernoulli_cpu(self):
        dist1 = self.make_bernoulli_dist()
        dist2 = self.make_bernoulli_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_bernoulli_bernoulli_gpu(self):
        dist1 = self.make_bernoulli_dist(True)
        dist2 = self.make_bernoulli_dist(True)
        self.check_kl(dist1, dist2)

    @testing.with_requires('scipy')
    def test_beta_beta_cpu(self):
        dist1 = self.make_beta_dist()
        dist2 = self.make_beta_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_beta_beta_gpu(self):
        dist1 = self.make_beta_dist(True)
        dist2 = self.make_beta_dist(True)
        self.check_kl(dist1, dist2)

    @testing.with_requires('numpy>=1.11')
    def test_categorical_categorical_cpu(self):
        dist1 = self.make_categorical_dist()
        dist2 = self.make_categorical_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_categorical_categorical_gpu(self):
        dist1 = self.make_categorical_dist(True)
        dist2 = self.make_categorical_dist(True)
        self.check_kl(dist1, dist2)

    @testing.with_requires('scipy')
    def test_dirichlet_dirichlet_cpu(self):
        dist1 = self.make_dirichlet_dist()
        dist2 = self.make_dirichlet_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_dirichlet_dirichlet_gpu(self):
        dist1 = self.make_dirichlet_dist(True)
        dist2 = self.make_dirichlet_dist(True)
        self.check_kl(dist1, dist2)

    def test_exponential_exponential_cpu(self):
        dist1 = self.make_exponential_dist()
        dist2 = self.make_exponential_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_exponential_exponential_gpu(self):
        dist1 = self.make_exponential_dist(True)
        dist2 = self.make_exponential_dist(True)
        self.check_kl(dist1, dist2)

    @testing.with_requires('scipy')
    def test_gamma_gamma_cpu(self):
        dist1 = self.make_gamma_dist()
        dist2 = self.make_gamma_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_gamma_gamma_gpu(self):
        dist1 = self.make_gamma_dist(True)
        dist2 = self.make_gamma_dist(True)
        self.check_kl(dist1, dist2)

    def test_geometric_geometric_cpu(self):
        dist1 = self.make_geometric_dist()
        dist2 = self.make_geometric_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_geometric_geometric_gpu(self):
        dist1 = self.make_geometric_dist(True)
        dist2 = self.make_geometric_dist(True)
        self.check_kl(dist1, dist2)

    @testing.with_requires('scipy')
    def test_gumbel_gumbel_cpu(self):
        dist1 = self.make_gumbel_dist()
        dist2 = self.make_gumbel_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_gumbel_gumbel_gpu(self):
        dist1 = self.make_gumbel_dist(True)
        dist2 = self.make_gumbel_dist(True)
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

    def test_log_normal_log_normal_cpu(self):
        dist1 = self.make_log_normal_dist()
        dist2 = self.make_log_normal_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_log_normal_log_normal_gpu(self):
        dist1 = self.make_log_normal_dist(True)
        dist2 = self.make_log_normal_dist(True)
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

    def test_one_hot_categorical_one_hot_categorical_cpu(self):
        dist1 = self.make_one_hot_categorical_dist()
        dist2 = self.make_one_hot_categorical_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_one_hot_categorical_one_hot_categorical_gpu(self):
        dist1 = self.make_one_hot_categorical_dist(True)
        dist2 = self.make_one_hot_categorical_dist(True)
        self.check_kl(dist1, dist2)

    def test_pareto_pareto_cpu(self):
        dist1 = self.make_pareto_dist()
        dist2 = self.make_pareto_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_pareto_pareto_gpu(self):
        dist1 = self.make_pareto_dist(True)
        dist2 = self.make_pareto_dist(True)
        self.check_kl(dist1, dist2)

    @testing.with_requires('scipy')
    def test_poisson_poisson_cpu(self):
        dist1 = self.make_poisson_dist()
        dist2 = self.make_poisson_dist()
        self.check_kl(dist1, dist2)

    @attr.gpu
    def test_poisson_poisson_gpu(self):
        dist1 = self.make_poisson_dist(True)
        dist2 = self.make_poisson_dist(True)
        self.check_kl(dist1, dist2)

    def test_uniform_uniform_cpu(self):
        for use_loc_scale1 in [True, False]:
            for use_loc_scale2 in [True, False]:
                dist1 = self.make_uniform_dist(use_loc_scale=use_loc_scale1)
                dist2 = self.make_uniform_dist(use_loc_scale=use_loc_scale2)
                self.check_kl(dist1, dist2)

    @attr.gpu
    def test_uniform_uniform_gpu(self):
        for use_loc_scale1 in [True, False]:
            for use_loc_scale2 in [True, False]:
                dist1 = self.make_uniform_dist(
                    True, use_loc_scale=use_loc_scale1)
                dist2 = self.make_uniform_dist(
                    True, use_loc_scale=use_loc_scale2)
                self.check_kl(dist1, dist2)


testing.run_module(__name__, __file__)

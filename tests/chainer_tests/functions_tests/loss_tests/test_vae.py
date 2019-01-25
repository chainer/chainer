import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(
    *testing.product({
        'wrap_m': [True, False],
        'wrap_v': [True, False],
        'reduce': ['no', 'sum', 'mean']
    })
)
class TestGaussianKLDivergence(unittest.TestCase):

    def setUp(self):
        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.ln_var = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

        # Refer to Appendix B in the original paper
        # Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114)
        loss = -(1 + self.ln_var -
                 self.mean * self.mean -
                 numpy.exp(self.ln_var)) * 0.5
        if self.reduce == 'sum':
            self.expect = numpy.sum(loss)
        elif self.reduce == 'mean':
            self.expect = numpy.mean(loss)
        elif self.reduce == 'no':
            self.expect = loss

    def check_gaussian_kl_divergence(self, mean, ln_var):
        if self.wrap_m:
            mean = chainer.Variable(mean)
        if self.wrap_v:
            ln_var = chainer.Variable(ln_var)
        actual = cuda.to_cpu(
            F.gaussian_kl_divergence(mean, ln_var, self.reduce).data)
        actual = cuda.to_cpu(
            F.gaussian_kl_divergence(mean, ln_var, self.reduce).data)
        testing.assert_allclose(self.expect, actual)

    @condition.retry(3)
    def test_gaussian_kl_divergence_cpu(self):
        self.check_gaussian_kl_divergence(self.mean, self.ln_var)

    @attr.gpu
    @condition.retry(3)
    def test_gaussian_kl_divergence_gpu(self):
        self.check_gaussian_kl_divergence(cuda.to_gpu(self.mean),
                                          cuda.to_gpu(self.ln_var))


class TestGaussianKLDivergenceInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.ln_var = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        m = chainer.Variable(xp.asarray(self.mean))
        v = chainer.Variable(xp.asarray(self.ln_var))
        with self.assertRaises(ValueError):
            F.gaussian_kl_divergence(m, v, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


@testing.parameterize(
    *testing.product({
        'wrap_x': [True, False],
        'wrap_y': [True, False],
        'reduce': ['no', 'sum', 'mean']
    })
)
class TestBernoulliNLL(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

        # Refer to Appendix C.1 in the original paper
        # Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114)
        p = 1 / (1 + numpy.exp(-self.y))
        self.expect = -(self.x * numpy.log(p) +
                        (1 - self.x) * numpy.log(1 - p))
        if self.reduce == 'sum':
            self.expect = numpy.sum(self.expect)
        elif self.reduce == 'mean':
            self.expect = numpy.mean(self.expect)

    def check_bernoulli_nll(self, x, y):
        if self.wrap_x:
            x = chainer.Variable(x)
        if self.wrap_y:
            y = chainer.Variable(y)
        actual = cuda.to_cpu(F.bernoulli_nll(x, y, self.reduce).data)
        testing.assert_allclose(self.expect, actual)

    @condition.retry(3)
    def test_bernoulli_nll_cpu(self):
        self.check_bernoulli_nll(self.x, self.y)

    @attr.gpu
    @condition.retry(3)
    def test_bernoulli_nll_gpu(self):
        self.check_bernoulli_nll(cuda.to_gpu(self.x),
                                 cuda.to_gpu(self.y))


class TestBernoulliNLLInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        y = chainer.Variable(xp.asarray(self.y))
        with self.assertRaises(ValueError):
            F.bernoulli_nll(x, y, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


@testing.parameterize(
    *testing.product({
        'wrap_x': [True, False],
        'wrap_m': [True, False],
        'wrap_v': [True, False],
        'reduce': ['no', 'sum', 'mean']
    })
)
class TestGaussianNLL(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.ln_var = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

        # Refer to Appendix C.2 in the original paper
        # Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114)
        x_d = self.x - self.mean
        var = numpy.exp(self.ln_var)

        self.expect = (0.5 * numpy.log(2 * numpy.pi) +
                       0.5 * self.ln_var +
                       x_d * x_d / var * 0.5)
        if self.reduce == 'sum':
            self.expect = numpy.sum(self.expect)
        elif self.reduce == 'mean':
            self.expect = numpy.mean(self.expect)

    def check_gaussian_nll(self, x, mean, ln_var):
        if self.wrap_x:
            x = chainer.Variable(x)
        if self.wrap_m:
            mean = chainer.Variable(mean)
        if self.wrap_v:
            ln_var = chainer.Variable(ln_var)
        actual = cuda.to_cpu(F.gaussian_nll(x, mean, ln_var, self.reduce).data)
        testing.assert_allclose(self.expect, actual)

    @condition.retry(3)
    def test_gaussian_nll_cpu(self):
        self.check_gaussian_nll(self.x, self.mean, self.ln_var)

    @attr.gpu
    @condition.retry(3)
    def test_gaussian_nll_gpu(self):
        self.check_gaussian_nll(cuda.to_gpu(self.x),
                                cuda.to_gpu(self.mean),
                                cuda.to_gpu(self.ln_var))


class TestGaussianNLLInvalidReductionOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.mean = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)
        self.ln_var = numpy.random.uniform(-1, 1, (3,)).astype(numpy.float32)

    def check_invalid_option(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        mean = chainer.Variable(xp.asarray(self.mean))
        ln_var = chainer.Variable(xp.asarray(self.ln_var))

        with self.assertRaises(ValueError):
            F.gaussian_nll(x, mean, ln_var, 'invalid_option')

    def test_invalid_option_cpu(self):
        self.check_invalid_option(numpy)

    @attr.gpu
    def test_invalid_option_gpu(self):
        self.check_invalid_option(cuda.cupy)


testing.run_module(__name__, __file__)

import unittest

import numpy

from chainer.backends import cuda
from chainer import distributions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2), (1,)],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'extreme_values': [True, False],
    'binary_check': [True, False],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestBernoulli(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Bernoulli
        self.scipy_dist = stats.bernoulli
        self.options = {'binary_check': self.binary_check}

        self.test_targets = set([
            'batch_shape', 'entropy', 'log_prob', 'mean', 'prob', 'sample',
            'stddev', 'support', 'variance'])

        if self.extreme_values:
            p = numpy.random.randint(0, 2, self.shape).astype(numpy.float32)
        else:
            p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)

        self.params = {'p': p}
        self.scipy_params = {'p': p}

        self.support = '{0, 1}'
        self.continuous = False

        self.old_settings = None
        if self.extreme_values:
            self.old_settings = numpy.seterr(divide='ignore', invalid='ignore')

    def tearDown(self):
        if self.old_settings is not None:
            numpy.seterr(**self.old_settings)

    def sample_for_test(self):
        smp = numpy.random.randint(
            2, size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp

    def sample_for_binary_check_test(self):
        smp = numpy.random.uniform(
            low=0.1, high=0.9,
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp

    def check_log_prob_binary_check(self, is_gpu):
        smp = self.sample_for_binary_check_test()
        if is_gpu:
            log_prob = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob = self.cpu_dist.log_prob(smp).data
        xp = cuda.get_array_module(log_prob)
        if self.binary_check:
            self.assertTrue(xp.all(log_prob == -xp.inf))
        else:
            self.assertTrue(xp.all(xp.isfinite(log_prob)))

    def test_log_prob_binary_check_cpu(self):
        self.check_log_prob_binary_check(False)

    @attr.gpu
    def test_log_prob_binary_check_gpu(self):
        self.check_log_prob_binary_check(True)

    def check_prob_binary_check(self, is_gpu):
        smp = self.sample_for_binary_check_test()
        if is_gpu:
            prob = self.gpu_dist.prob(cuda.to_gpu(smp)).data
        else:
            prob = self.cpu_dist.prob(smp).data
        xp = cuda.get_array_module(prob)
        if self.binary_check:
            self.assertTrue(xp.all(prob == 0))
        else:
            self.assertTrue(xp.all(prob > 0))

    def test_prob_binary_check_cpu(self):
        self.check_prob_binary_check(False)

    @attr.gpu
    def test_prob_binary_check_gpu(self):
        self.check_prob_binary_check(True)


@testing.parameterize(*testing.product({
    'logit_shape,x_shape': [
        [(2, 3), (2, 3)],
        [(), ()],
        [(), (3,)]
    ],
    'dtype': [numpy.float32, numpy.float64],
}))
class TestBernoulliLogProb(unittest.TestCase):

    def setUp(self):
        self.logit = numpy.random.normal(
            size=self.logit_shape).astype(self.dtype)
        self.x = numpy.random.randint(
            0, 2, size=self.x_shape).astype(self.dtype)
        self.gy = numpy.random.normal(size=self.x_shape).astype(self.dtype)
        self.ggx = numpy.random.normal(
            size=self.logit_shape).astype(self.dtype)
        self.backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def check_forward(self, logit_data, x_data):
        distributions.bernoulli._bernoulli_log_prob(logit_data, x_data)

    def test_forward_cpu(self):
        self.check_forward(self.logit, self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.logit), cuda.to_gpu(self.x))

    def check_backward(self, logit_data, x_data, y_grad):
        def f(logit):
            return distributions.bernoulli._bernoulli_log_prob(
                logit, x_data)
        gradient_check.check_backward(
            f, logit_data, y_grad, **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.logit, self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.logit), cuda.to_gpu(self.x),
                            cuda.to_gpu(self.gy))

    def check_double_backward(self, logit_data, x_data, y_grad, x_grad_grad):
        def f(logit):
            return distributions.bernoulli._bernoulli_log_prob(
                logit, x_data)
        gradient_check.check_double_backward(
            f, logit_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **self.backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.logit, self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.logit), cuda.to_gpu(self.x),
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    def test_backward_where_logit_has_infinite_values(self):
        self.logit[...] = numpy.inf
        with numpy.errstate(invalid='ignore'):
            log_prob = distributions.bernoulli._bernoulli_log_prob(
                self.logit, self.x)

        # just confirm that the backward method runs without raising error.
        log_prob.backward()


testing.run_module(__name__, __file__)

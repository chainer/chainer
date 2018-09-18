import numpy

from chainer import cuda
from chainer import distributions
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'k': [3],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestLogNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Multinomial
        self.scipy_dist = stats.multinomial

        self.test_targets = set([
            "batch_shape", "event_shape",
            "sample"])

        n = numpy.random.randint(1, 30, self.shape).astype(numpy.int32)
        p = numpy.random.normal(
            size=self.shape+(self.k,)).astype(numpy.float32)
        p = numpy.exp(p)
        p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
        self.n = n
        self.p = p
        self.params = {"n": n, "p": p}
        self.scipy_params = {"n": n, "p": p}

        self.continuous = False
        self.event_shape = (self.k,)

    def check_log_prob(self, is_gpu):
        obo_p = self.p.reshape(-1, self.k)
        obo_n = self.n.reshape(-1)
        smp = [numpy.random.multinomial(one_n, one_p, size=1)
               for one_n, one_p in zip(obo_n, obo_p)]
        smp = numpy.stack(smp)
        smp = smp.reshape(self.shape + (-1,))
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp).data

        obo_smp = smp.reshape(-1, self.k)
        log_prob2 = [self.scipy_dist.logpmf(one_smp, one_n, one_p)
                     for one_smp, one_n, one_p in zip(obo_smp, obo_n, obo_p)]
        log_prob2 = numpy.stack(log_prob2).reshape(self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @testing.attr.gpu
    def test_log_prob_gpu(self):
        self.check_log_prob(True)


testing.run_module(__name__, __file__)

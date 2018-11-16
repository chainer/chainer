from chainer import cuda
from chainer import distributions
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
    'k': [3],
}))
@testing.fix_random()
@testing.with_requires('scipy>=0.19')
class TestOneHotCategorical(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.OneHotCategorical
        self.scipy_dist = stats.multinomial

        self.test_targets = set([
            "batch_shape", "event_shape", "mean", "sample"])

        n = numpy.ones(self.shape).astype(numpy.int32)
        p = numpy.random.normal(
            size=self.shape+(self.k,)).astype(numpy.float32)
        p = numpy.exp(p)
        p /= numpy.expand_dims(p.sum(axis=-1), axis=-1)
        self.n, self.p = n, p
        self.params = {"p": p}
        self.scipy_params = {"n": n, "p": p}

        self.continuous = False
        self.event_shape = (self.k,)

    def sample_for_test(self):
        obo_p = self.p.reshape(-1, self.k)
        obo_n = self.n.reshape(-1)
        smp = [numpy.random.multinomial(one_n, one_p, size=self.sample_shape)
               for one_n, one_p in zip(obo_n, obo_p)]
        smp = numpy.stack(smp)
        smp = numpy.moveaxis(smp, 0, -2)
        smp = smp.reshape(self.sample_shape + self.shape + (self.k,))
        return smp

    def check_log_prob(self, is_gpu):
        smp = self.sample_for_test()
        if is_gpu:
            log_prob1 = self.gpu_dist.log_prob(cuda.to_gpu(smp)).data
        else:
            log_prob1 = self.cpu_dist.log_prob(smp).data

        onebyone_smp = smp.reshape(self.sample_shape + (-1,) + (self.k,))
        onebyone_smp = numpy.moveaxis(onebyone_smp, -2, 0)
        onebyone_smp = onebyone_smp.reshape(
            (-1,) + self.sample_shape + (self.k,))
        log_prob2 = []
        for one_params, one_smp in zip(
                self.scipy_onebyone_params_iter(), onebyone_smp):
            log_prob2.append(self.scipy_dist.logpmf(one_smp, **one_params))
        log_prob2 = numpy.stack(log_prob2)
        log_prob2 = numpy.moveaxis(log_prob2, 0, -1)
        log_prob2 = log_prob2.reshape(self.sample_shape + self.shape)
        testing.assert_allclose(log_prob1, log_prob2)

    def test_log_prob_cpu(self):
        self.check_log_prob(False)

    @testing.attr.gpu
    def test_log_prob_gpu(self):
        self.check_log_prob(True)


testing.run_module(__name__, __file__)

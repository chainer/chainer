import numpy 

from chainer import distributions
from chainer import testing


@testing.parameterize(*testing.product({
	'shape': [(2, 3), ()],
	'is_variable': [True, False],
	'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestBinomial(testing.distribution_unittest):

	scipy_onebyone = True

	def setUp_configure(self):
		from scipy import stats
		self.dist = distributions.Binomial
		self.scipy_dist = stats.binom

		self.test_targets = set([
			"batch_shape", "mean", "variance",
			"support"])

		p = numpy.random.uniform(0, 1, self.shape).astype(numpy.float32)
		n = numpy.random.randint(1, 10, self.shape).astype(numpy.float32)
		self.params = {"n": n, "p": p}
		self.scipy_params = {"n": n, "p": p}

		self.support = 'positive integer'
		self.continuous = False

	def sample_for_test(self):
		smp = numpy.random.binomial(
			0.5, 10, self.sample_shape + self.shape).astype(numpy.int32)
		return smp

testing.run_module(__name__, __file__)

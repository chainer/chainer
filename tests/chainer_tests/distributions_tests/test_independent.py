import functools
import itertools
import operator

import numpy

from chainer import distributions
from chainer import testing
from chainer.testing import array
from chainer.testing import attr
from chainer import utils


def skip_not_in_params(property):
    def decorator(f):
        @functools.wraps(f)
        def new_f(self, *args, **kwargs):
            if property not in self.params.keys():
                self.skipTest(
                    "\'%s\' does not exist in params.keys()." % property)
            else:
                f(self, *args, **kwargs)
        return new_f
    return decorator


def _generate_valid_shape_pattern(
        inner_shape, inner_event_shape, reinterpreted_batch_ndims):
    shape_pattern = []
    for bs, es, m in itertools.product(
            inner_shape, inner_event_shape, reinterpreted_batch_ndims):
        if (m is not None) and (m > len(bs)):
            continue
        shape_pattern.append({
            'full_shape': bs + es,
            'inner_shape': bs,
            'inner_event_shape': es,
            'reinterpreted_batch_ndims': m
        })
    return shape_pattern


def _generate_test_parameter(
        parameter_list, inner_shape, inner_event_shape,
        reinterpreted_batch_ndims):
    shape_pattern = _generate_valid_shape_pattern(
        inner_shape, inner_event_shape, reinterpreted_batch_ndims)
    return [
        dict(dicts[0], **dicts[1])
        for dicts in itertools.product(parameter_list, shape_pattern)
    ]


@testing.parameterize(*_generate_test_parameter(
    testing.product({
        'sample_shape': [(3, 2), ()],
        'is_variable': [True, False]
    }),
    inner_shape=[(4, 5), (5,), ()],
    inner_event_shape=[()],
    reinterpreted_batch_ndims=[1, 0, None]
))
@testing.fix_random()
@testing.with_requires('scipy')
class TestIndependentNormal(testing.distribution_unittest):

    scipy_onebyone = True

    def _build_inner_distribution(self):
        pass

    def setUp_configure(self):
        from scipy import stats
        self.dist = lambda **params: distributions.Independent(
            distributions.Normal(**params), self.reinterpreted_batch_ndims)

        self.test_targets = set([
            "batch_shape", "entropy", "event_shape", "log_prob",
            "support"])

        loc = utils.force_array(numpy.random.uniform(
            -1, 1, self.full_shape).astype(numpy.float32))
        scale = utils.force_array(numpy.exp(numpy.random.uniform(
            -1, 1, self.full_shape)).astype(numpy.float32))

        if self.reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = max(0, len(self.inner_shape) - 1)
        else:
            reinterpreted_batch_ndims = self.reinterpreted_batch_ndims

        batch_ndim = len(self.inner_shape) - reinterpreted_batch_ndims
        self.shape = self.inner_shape[:batch_ndim]
        self.event_shape = \
            self.inner_shape[batch_ndim:] + self.inner_event_shape
        d = functools.reduce(operator.mul, self.event_shape, 1)

        if self.event_shape == ():
            self.scipy_dist = stats.norm

            self.params = {"loc": loc, "scale": scale}
            self.scipy_params = {"loc": loc, "scale": scale}

        else:
            self.scipy_dist = stats.multivariate_normal

            scale_tril = numpy.eye(d).astype(numpy.float32) * \
                scale.reshape(self.shape + (d,))[..., None]
            cov = numpy.einsum('...ij,...jk->...ik', scale_tril, scale_tril)

            self.params = {"loc": loc, "scale": scale}
            self.scipy_params = {"mean": numpy.reshape(
                loc, self.shape + (d,)), "cov": cov}

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.full_shape
        ).astype(numpy.float32)
        return smp

    def test_batch_ndim_error(self):
        with self.assertRaises(ValueError):
            distributions.Independent(
                distributions.Normal(**self.params),
                len(self.inner_shape) + 1)

    def check_covariance(self, is_gpu):
        if is_gpu:
            cov1 = self.gpu_dist.covariance.array
        else:
            cov1 = self.cpu_dist.covariance.array
        cov2 = self.params['cov']
        array.assert_allclose(cov1, cov2)

    @skip_not_in_params('cov')
    def test_covariance_cpu(self):
        self.check_covariance(False)

    @skip_not_in_params('cov')
    @attr.gpu
    def test_covariance_gpu(self):
        self.check_covariance(True)


testing.run_module(__name__, __file__)

import chainer
from chainer import functions as F
import numpy

import chainerx

from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


_in_out_loss_dtypes = dtype_utils._permutate_dtype_mapping([
    (('float16', 'float16'), 'float16'),
    (('float32', 'float32'), 'float32'),
    (('float64', 'float64'), 'float64'),
    (('float32', 'float16'), 'float32'),
    (('float64', 'float16'), 'float64'),
    (('float64', 'float32'), 'float64'),
])


class LossBase(op_utils.ChainerOpTest):

    def generate_inputs(self):
        y = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
        return y, targ

    def forward_chainerx(self, inputs):
        return self.forward_xp(inputs, chainerx)

    def forward_chainer(self, inputs):
        return self.forward_xp(inputs, F)

    def forward_xp(self, inputs, xp):
        raise NotImplementedError(
            'Op test implementation must override `forward_xp`.')


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'shape', [
                (2, 2),
                (3, 3, 3),
                (5, 5, 5),
                (4, 1, 2, 4)
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', _in_out_loss_dtypes)
    ])
))
class TestSquaredError(LossBase):

    def forward_xp(self, inputs, xp):
        x1, x2 = inputs
        return xp.squared_error(x1, x2),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'shape', [
                (2, 2),
                (3, 3, 3),
                (5, 5, 5),
                (4, 1, 2, 4)
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', _in_out_loss_dtypes)
    ])
))
class TestAbsoluteError(LossBase):

    # Absolute is non-differentiable at zero.
    dodge_nondifferentiable = True

    def forward_xp(self, inputs, xp):
        x1, x2 = inputs
        return xp.absolute_error(x1, x2),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'shape', [
                (2, 2),
                (3, 3, 3),
                (5, 5, 5),
                (4, 1, 2, 4)
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', _in_out_loss_dtypes)
    ])
))
class TestGaussianKLDivergence(LossBase):

    def forward_xp(self, inputs, xp):
        mean, ln_var = inputs
        if xp is chainerx:
            out = xp.gaussian_kl_divergence(mean, ln_var)
        else:
            out = xp.gaussian_kl_divergence(mean, ln_var, reduce='no')
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'shape', [
                (2, 2),
                (3, 3, 3),
                (5, 5, 5),
                (4, 1, 2, 4)
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', _in_out_loss_dtypes),
        chainer.testing.from_pytest_parameterize(
            'delta', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    ])
))
class TestHuberLoss(LossBase):

    def generate_inputs(self):
        x, t = super().generate_inputs()
        mask = numpy.abs(numpy.abs(x - t) - self.delta) > 1e-3
        return x * mask, t * mask

    def forward_xp(self, inputs, xp):
        x, t = inputs
        if xp is chainerx:
            out = xp.huber_loss(x, t, self.delta)
        else:
            out = xp.huber_loss(x, t, self.delta, reduce='no')
        return out,

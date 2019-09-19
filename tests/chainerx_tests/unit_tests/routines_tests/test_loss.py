import numpy

import chainer
from chainer import functions as F
import chainerx
from chainerx_tests import op_utils


_loss_shapes = [
    (2, 2),
    (3, 3, 3),
    (5, 5, 5),
    (4, 1, 2, 4),
]


class LossBase(op_utils.ChainerOpTest):

    def setup(self):
        super().setup()
        if self.in_dtype == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 3e-1})

    def generate_inputs(self):
        y = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
        return y.astype(self.in_dtype), targ.astype(self.in_dtype)

    def forward_chainerx(self, inputs):
        out, = self.forward_xp(inputs, chainerx)
        return out,

    def forward_chainer(self, inputs):
        return self.forward_xp(inputs, F)

    def forward_xp(self, inputs, xp):
        raise NotImplementedError(
            'Op test implementation must override `forward_xp`.')


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': _loss_shapes,
        'in_dtype': chainerx.testing.float_dtypes,
    })
))
class TestSquaredError(LossBase):

    def forward_xp(self, inputs, xp):
        x1, x2 = inputs
        return xp.squared_error(x1, x2),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': _loss_shapes,
        'in_dtype': chainerx.testing.float_dtypes,
    })
))
class TestAbsoluteError(LossBase):

    # Absolute is non-differentiable at zero.
    dodge_nondifferentiable = True

    def forward_xp(self, inputs, xp):
        x1, x2 = inputs
        return xp.absolute_error(x1, x2),


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': _loss_shapes,
        'in_dtype': chainerx.testing.float_dtypes,
    })
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
    chainer.testing.product({
        'shape': _loss_shapes,
        'in_dtype': chainerx.testing.float_dtypes,
        'delta': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    })
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


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': _loss_shapes,
        'in_dtype': chainerx.testing.float_dtypes,
        't_dtype': ['int8', 'int16', 'int32', 'int64'],
    })
))
class TestSigmoidCrossEntropy(LossBase):

    def generate_inputs(self):
        x = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
        self.t = targ.astype(self.t_dtype)
        return x.astype(self.in_dtype),

    def forward_xp(self, inputs, xp):
        x, = inputs
        # TODO(aksub99): Improve implementation to avoid non-differentiability
        # wrt targets
        if xp is chainerx:
            t = self.backend_config.get_array(self.t)
            out = xp.sigmoid_cross_entropy(x, t)
        else:
            t = self.t
            out = xp.sigmoid_cross_entropy(x, t, normalize=False, reduce='no')
        return out,

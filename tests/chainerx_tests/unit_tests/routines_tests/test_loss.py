import numpy

import chainer
from chainer import functions as F
import chainerx

from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


_loss_shapes = [
    (2, 2),
    (3, 3, 3),
    (5, 5, 5),
    (4, 1, 2, 4),
]


_in_out_loss_dtypes = dtype_utils._permutate_dtype_mapping([
    (('float16', 'float16'), 'float16'),
    (('float32', 'float32'), 'float32'),
    (('float64', 'float64'), 'float64'),
    (('float32', 'float16'), 'float32'),
    (('float64', 'float16'), 'float64'),
    (('float64', 'float32'), 'float64'),
])


class LossBase(op_utils.ChainerOpTest):

    def setup(self):
        super().setup()
        in_dtype1, in_dtype2 = self.in_dtypes
        if in_dtype1 == 'float16' or in_dtype2 == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 3e-1})

    def generate_inputs(self):
        in_dtype1, in_dtype2 = self.in_dtypes
        y = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
        return y.astype(in_dtype1), targ.astype(in_dtype2)

    def forward_chainerx(self, inputs):
        out, = self.forward_xp(inputs, chainerx)
        return out,

    def forward_chainer(self, inputs):
        dtype = numpy.result_type(*inputs)
        inputs = [x.astype(dtype) for x in inputs]
        output, = self.forward_xp(inputs, F)
        output.array = output.array.astype(self.out_dtype)
        return output,

    def forward_xp(self, inputs, xp):
        raise NotImplementedError(
            'Op test implementation must override `forward_xp`.')


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': _loss_shapes,
        'in_dtypes,out_dtype': _in_out_loss_dtypes,
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
        'in_dtypes,out_dtype': _in_out_loss_dtypes,
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
        'in_dtypes,out_dtype': _in_out_loss_dtypes,
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
        'in_dtypes,out_dtype': _in_out_loss_dtypes,
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
        'x_dtype': chainerx.testing.float_dtypes,
        't_dtype': ['int8', 'int16', 'int32', 'int64'],
    })
))
class TestSigmoidCrossEntropy(op_utils.ChainerOpTest):

    def setup(self):
        if self.x_dtype == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 3e-1})

    def generate_inputs(self):
        x = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
        self.t = targ.astype(self.t_dtype)
        return x.astype(self.x_dtype),

    def forward_chainerx(self, inputs):
        x, = inputs
        # TODO(aksub99): Improve implementation to avoid non-differentiability
        # wrt targets
        t = self.backend_config.get_array(self.t)
        out = chainerx.sigmoid_cross_entropy(x, t)
        return out,

    def forward_chainer(self, inputs):
        x, = inputs
        t = self.t
        out = F.sigmoid_cross_entropy(x, t, normalize=False, reduce='no')
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'x_dtype': chainerx.testing.float_dtypes,
        't_dtype': chainerx.testing.signed_integral_dtypes,
    })
))
class TestSoftmaxCrossEntropy(op_utils.ChainerOpTest):

    def setup(self):
        self.shape = (2, 2)

        t_shape = self.shape[0],
        t = numpy.random.randint(0, self.shape[1], t_shape)
        self.t = t.astype(self.t_dtype)

        if self.x_dtype == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 3e-1})

    def generate_inputs(self):
        x = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        return x.astype(self.x_dtype),

    def forward_chainerx(self, inputs):
        x, = inputs
        t = self.backend_config.get_array(self.t)
        out = chainerx.softmax_cross_entropy(x, t)
        return out,

    def forward_chainer(self, inputs):
        x, = inputs
        out = F.softmax_cross_entropy(x, self.t, reduce='no')
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(2, 2), (3, 5), (7, 1)],
        'x_dtype': chainerx.testing.float_dtypes,
        't_dtype': ['int8', 'int16', 'int32', 'int64'],
        'norm_float,norm_str': [(1.0, 'L1'), (2.0, 'L2')],
    })
))
class TestHinge(op_utils.ChainerOpTest):

    dodge_nondifferentiable = True

    def setup(self):
        if self.x_dtype == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-3})
            self.check_backward_options.update({'rtol': 1e-2, 'atol': 5e-3})
            self.check_double_backward_options.update(
                {'rtol': 1e-2, 'atol': 3e-1})

    def generate_inputs(self):
        n, k = self.shape
        x = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        self.t = numpy.random.randint(k, size=n).astype(self.t_dtype)
        return x.astype(self.x_dtype),

    def forward_chainerx(self, inputs):
        x, = inputs
        t = self.backend_config.get_array(self.t)
        norm = self.norm_float
        out = chainerx.hinge(x, t, norm=norm)
        return out,

    def forward_chainer(self, inputs):
        x, = inputs
        t = self.t
        norm = self.norm_str
        out = F.hinge(x, t, norm=norm, reduce='no')
        return out,

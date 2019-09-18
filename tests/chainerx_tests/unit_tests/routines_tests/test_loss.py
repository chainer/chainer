import numpy
import pytest

import chainer
from chainer import functions as F
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


_loss_dtype_error = [
    ('float16', 'float32'),
    ('float16', 'float64'),
    ('float32', 'float16'),
    ('float32', 'float64'),
    ('float64', 'float16'),
    ('float64', 'float32'),
    ('bool_', 'bool_'),
    ('int64', 'int64'),
    ('uint8', 'uint8'),
    ('float32', 'int32'),
    ('int16', 'float32'),
]


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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype1,dtype2', _loss_dtype_error)
def test_squared_error_invalid_dtypes(device, dtype1, dtype2):
    shape = (3, 2)
    x1 = chainerx.ones(shape, dtype=dtype1)
    x2 = chainerx.ones(shape, dtype=dtype2)
    with pytest.raises(chainerx.DtypeError):
        chainerx.squared_error(x1, x2)


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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype1,dtype2', _loss_dtype_error)
def test_absolute_error_invalid_dtypes(device, dtype1, dtype2):
    shape = (3, 2)
    x1 = chainerx.ones(shape, dtype=dtype1)
    x2 = chainerx.ones(shape, dtype=dtype2)
    with pytest.raises(chainerx.DtypeError):
        chainerx.absolute_error(x1, x2)


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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype1,dtype2', _loss_dtype_error)
def test_gaussian_kl_divergence_invalid_dtypes(device, dtype1, dtype2):
    shape = (3, 2)
    x1 = chainerx.ones(shape, dtype=dtype1)
    x2 = chainerx.ones(shape, dtype=dtype2)
    with pytest.raises(chainerx.DtypeError):
        chainerx.gaussian_kl_divergence(x1, x2)


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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype1,dtype2', _loss_dtype_error)
def test_huber_loss_invalid_dtypes(device, dtype1, dtype2):
    shape = (3, 2)
    x1 = chainerx.ones(shape, dtype=dtype1)
    x2 = chainerx.ones(shape, dtype=dtype2)
    with pytest.raises(chainerx.DtypeError):
        chainerx.huber_loss(x1, x2, 0.1)


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
    ])
))
class TestSigmoidCrossEntropy(LossBase):

    def generate_inputs(self):
        y = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
        self.t = targ
        return y,

    def forward_xp(self, inputs, xp):
        x, = inputs
        # TODO(aksub99): Improve implementation to avoid non-differentiability
        # wrt targets
        if xp is chainerx:
            t = self.backend_config.get_array(self.t)
            t = t.astype(numpy.int64)
            out = xp.sigmoid_cross_entropy(x, t)
        else:
            t = self.t.astype(numpy.int64)
            out = xp.sigmoid_cross_entropy(x, t, normalize=False, reduce='no')
        return out,


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('dtype1,dtype2', [
    ('float16', 'float16'),
    ('float32', 'float32'),
    ('float64', 'float64'),
    ('float16', 'float32'),
    ('float32', 'float64'),
    ('float64', 'float16'),
    ('float16', 'uint8'),
    ('float32', 'uint8'),
    ('float64', 'uint8'),
    ('float16', 'bool_'),
    ('float32', 'bool_'),
    ('float64', 'bool_'),
    ('bool_', 'bool_'),
    ('int32', 'int32'),
    ('uint8', 'uint8'),
    ('int32', 'float64'),
])
def test_sigmoid_cross_entropy_invalid_dtypes(device, dtype1, dtype2):
    shape = (3, 2)
    x1 = chainerx.ones(shape, dtype=dtype1)
    x2 = chainerx.ones(shape, dtype=dtype2)
    with pytest.raises(chainerx.DtypeError):
        chainerx.sigmoid_cross_entropy(x1, x2)

import unittest

import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import op_utils


def _create_batch_norm_ndarray_args(
        xp, device, x_shape, gamma_shape, beta_shape, mean_shape, var_shape,
        float_dtype):
    x = array_utils.create_dummy_ndarray(xp, x_shape, float_dtype)

    # Non-contiguous gamma and beta is not supported by CUDA.
    # TODO(hvy): Support non-contiguous gamma and beta with CUDA. Create a
    # contiguous copy in the cuDNN wrapper.
    pad_gamma_beta = device.backend.name != 'cuda'
    gamma = array_utils.create_dummy_ndarray(
        xp, gamma_shape, float_dtype, padding=pad_gamma_beta)
    beta = array_utils.create_dummy_ndarray(
        xp, beta_shape, float_dtype, padding=pad_gamma_beta)

    # Non-contiguous running values which are updated in-place are not
    # supported by CUDA, so we only pad for other devices.
    pad_running = device.backend.name != 'cuda'
    mean = array_utils.create_dummy_ndarray(
        xp, mean_shape, float_dtype, padding=pad_running)
    var = array_utils.create_dummy_ndarray(
        xp, var_shape, float_dtype, padding=pad_running, start=0)

    # TODO(imanishi): Remove them after supporting random test
    x /= x.size
    gamma /= gamma.size
    beta /= beta.size
    mean /= mean.size
    var /= var.size

    return x, gamma, beta, mean, var


# Note that CUDA (cuDNN) only supports batch normalization with 4 or
# 5-dimensional data. Arrays with smaller dimensions are supported by the
# CUDA backend, while those with larger dimensions are not.
# x_shape,reduced_shape,axis
_batch_norm_params = [
    ((3, 2), (2,), None),
    ((5, 4, 3, 2), (4, 3, 2), None),
    ((5, 4, 3, 2), (4, 3, 2), (0,)),
    ((5, 4, 3, 2), (4,), (0, 2, 3)),
    ((5, 4, 3, 2, 2), (4, 3, 2, 2), None),
    ((5, 4, 3, 2, 2), (4, 3, 2, 2), (0,)),
    ((5, 4, 3, 2, 2), (4,), (0, 2, 3, 4))
]


# x_shape,gamma_shape,beta_shape,mean_shape,var_shape,axis
_batch_norm_invalid_dimensions_params = [
    # Bad reduction, axis defaults to (0,) but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), None),
    # Bad reduction, axis is () but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), ()),
    # Bad reduction, axis is (2, 3) but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), (2, 3)),
    ((2, 3, 4, 5), (3, 4), (3,), (3,), (3,), (0, 2, 3)),  # Bad gamma shape.
    ((2, 3, 4, 5), (3,), (3, 4), (3,), (3,), (0, 2, 3)),  # Bad beta shape.
    ((2, 3, 4, 5), (3,), (3,), (3, 4), (3,), (0, 2, 3)),  # Bad mean shape.
    ((2, 3, 4, 5), (3,), (3,), (3,), (3, 4), (0, 2, 3)),  # Bad var shape.
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest(
    'x_shape,reduced_shape,axis', _batch_norm_params)
@chainer.testing.parameterize_pytest(
    'x_dtype', chainerx.testing.float_dtypes)
@chainer.testing.parameterize_pytest(
    'param_dtype', chainerx.testing.float_dtypes)
@chainer.testing.parameterize_pytest('eps', [2e-5, 5e-1])
@chainer.testing.parameterize_pytest('decay', [None, 0.5])
@chainer.testing.parameterize_pytest('contiguous', [None, 'C'])
class TestBatchNorm(op_utils.ChainerOpTest):

    def setup(self):
        reduced_shape = self.reduced_shape
        x_dtype = self.x_dtype
        param_dtype = self.param_dtype
        eps = self.eps
        decay = self.decay
        axis = self.axis
        contiguous = self.contiguous

        # - Non-contiguous running values which are updated in-place are not
        # supported by CUDA.
        # - Non-contiguous gamma and beta is not supported by CUDA.
        # TODO(hvy): Support non-contiguous gamma and beta with CUDA. Create a
        # contiguous copy in the cuDNN wrapper.
        if (chainerx.get_default_device().backend.name == 'cuda'
                and contiguous is None):
            raise unittest.SkipTest(
                'batch_norm with CUDA currently has limited support for '
                'non-contiguous inputs.')

        # BatchNorm is unstable for fp16 for both native and CUDA.
        # TODO(hvy): Fix backward and double backward for fp16.
        if x_dtype == 'float16' and param_dtype == 'float16':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        self.running_mean = numpy.random.uniform(
            -1, 1, reduced_shape).astype(param_dtype)
        self.running_var = numpy.random.uniform(
            0.1, 1, reduced_shape).astype(param_dtype)

        optional_args = {}
        if eps is not None:
            optional_args['eps'] = eps
        if decay is not None:
            optional_args['decay'] = decay
        if axis is not None:
            optional_args['axis'] = axis
        self.optional_args = optional_args

        # TODO(hvy): Fix forward, backward and double backward for fp16.
        if x_dtype == 'float16' or param_dtype == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
            self.check_double_backward_options.update({
                'rtol': 1e-1, 'atol': 1e-1})
        else:
            self.check_forward_options.update({
                'rtol': 1e-6, 'atol': 1e-5})
            self.check_backward_options.update({
                'rtol': 5e-3, 'atol': 5e-4})
            self.check_double_backward_options.update({
                'rtol': 5e-2, 'atol': 5e-3})

        # Running values that are recorded in forward for similarity checks.
        self.running_mean_chx = None
        self.running_var_chx = None
        self.running_mean_ch = None
        self.running_var_ch = None

    def generate_inputs(self):
        x_shape = self.x_shape
        reduced_shape = self.reduced_shape
        x_dtype = self.x_dtype
        param_dtype = self.param_dtype

        x = numpy.random.uniform(-1, 1, x_shape).astype(x_dtype)
        gamma = numpy.random.uniform(0.5, 1, reduced_shape).astype(param_dtype)
        beta = numpy.random.uniform(-1, 1, reduced_shape).astype(param_dtype)

        return x, gamma, beta,

    def forward_chainerx(self, inputs):
        x, gamma, beta = inputs

        running_mean = chainerx.array(self.running_mean, copy=True)
        running_var = chainerx.array(self.running_var, copy=True)

        y = chainerx.batch_norm(
            x, gamma, beta, running_mean=running_mean, running_var=running_var,
            **self.optional_args)

        # Record running values for later checks.
        self.running_mean_chx = running_mean
        self.running_var_chx = running_var

        return y,

    def forward_chainer(self, inputs):
        x, gamma, beta = inputs

        running_mean = self.running_mean.copy()
        running_var = self.running_var.copy()

        y = chainer.functions.batch_normalization(
            x, gamma, beta, running_mean=running_mean, running_var=running_var,
            **self.optional_args)

        # Record running values for later checks.
        self.running_mean_ch = running_mean
        self.running_var_ch = running_var

        return y,

    def check_forward_outputs(self, outputs, expected_outputs):
        super().check_forward_outputs(outputs, expected_outputs)

        # Check that running values are updated.
        if (self.x_dtype == 'float16'
                or self.param_dtype == 'float16'):
            check_running_options = {'rtol': 1e-1, 'atol': 1e-1}
        else:
            check_running_options = {'rtol': 1e-6, 'atol': 1e-5}

        chainerx.testing.assert_allclose(
            self.running_mean_chx, self.running_mean_ch,
            **check_running_options)
        chainerx.testing.assert_allclose(
            self.running_var_chx, self.running_var_ch, **check_running_options)


@pytest.mark.parametrize(
    'x_shape,gamma_shape,beta_shape,running_mean_shape,running_var_shape,axis',
    _batch_norm_invalid_dimensions_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_batch_norm_invalid_dimensions(
        device, x_shape, gamma_shape, beta_shape, running_mean_shape,
        running_var_shape, axis, float_dtype):
    x, gamma, beta, running_mean, running_var = (
        _create_batch_norm_ndarray_args(
            chainerx, device, x_shape, gamma_shape, beta_shape,
            running_mean_shape, running_var_shape, float_dtype))

    with pytest.raises(chainerx.DimensionError):
        chainerx.batch_norm(
            x, gamma, beta, running_mean=running_mean, running_var=running_var,
            eps=1e-2, decay=0.9, axis=axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest(
    'x_shape,reduced_shape,axis', _batch_norm_params)
@chainer.testing.parameterize_pytest(
    'x_dtype', chainerx.testing.float_dtypes)
@chainer.testing.parameterize_pytest(
    'param_dtype', chainerx.testing.float_dtypes)
@chainer.testing.parameterize_pytest('eps', [None, 3e-5, 1.2])
@chainer.testing.parameterize_pytest('contiguous', [None, 'C'])
class TestFixedBatchNorm(op_utils.ChainerOpTest):

    # Backward and double backward for fixed_batch_norm is not supported yet.
    skip_backward_test = True
    skip_double_backward_test = True

    def setup(self, float_dtype):
        x_dtype = self.x_dtype
        param_dtype = self.param_dtype
        eps = self.eps
        axis = self.axis

        optional_args = {}
        if eps is not None:
            optional_args['eps'] = eps
        if axis is not None:
            optional_args['axis'] = axis
        self.optional_args = optional_args

        if x_dtype == 'float16' or param_dtype == 'float16':
            self.check_forward_options.update({'rtol': 1e-1, 'atol': 1e-1})
        else:
            self.check_forward_options.update({'rtol': 1e-6, 'atol': 1e-5})

    def generate_inputs(self):
        x_shape = self.x_shape
        reduced_shape = self.reduced_shape
        x_dtype = self.x_dtype
        param_dtype = self.param_dtype

        x = numpy.random.uniform(-1, 1, x_shape).astype(x_dtype)
        gamma = numpy.random.uniform(-1, 1, reduced_shape).astype(param_dtype)
        beta = numpy.random.uniform(-1, 1, reduced_shape).astype(param_dtype)
        mean = numpy.random.uniform(-1, 1, reduced_shape).astype(param_dtype)
        var = numpy.random.uniform(0.1, 1, reduced_shape).astype(param_dtype)

        return x, gamma, beta, mean, var

    def forward_chainerx(self, inputs):
        x, gamma, beta, mean, var = inputs

        y = chainerx.fixed_batch_norm(
            x, gamma, beta, mean=mean, var=var, **self.optional_args)
        return y,

    def forward_chainer(self, inputs):
        x, gamma, beta, mean, var = inputs

        y = chainer.functions.fixed_batch_normalization(
            x, gamma, beta, mean=mean, var=var, **self.optional_args)
        return y,


@pytest.mark.parametrize(
    'x_shape,gamma_shape,beta_shape,mean_shape,var_shape,axis',
    _batch_norm_invalid_dimensions_params)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fixed_batch_norm_invalid_dimensions(
        device, x_shape, gamma_shape, beta_shape, mean_shape, var_shape, axis,
        float_dtype):
    x, gamma, beta, mean, var = _create_batch_norm_ndarray_args(
        chainerx, device, x_shape, gamma_shape, beta_shape, mean_shape,
        var_shape, float_dtype)

    with pytest.raises(chainerx.DimensionError):
        chainerx.fixed_batch_norm(
            x, gamma, beta, mean=mean, var=var, eps=1e-2, axis=axis)

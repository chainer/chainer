import chainer
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils


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
# 5-dimenisional data.
# x_shape,reduced_shape,axis
_batch_norm_params = [
    ((2, 3, 4, 5), (3, 4, 5), None),
    ((2, 3, 4, 5), (3, 4, 5), (0,)),
    ((2, 3, 4, 5), (3,), (0, 2, 3)),
    ((2, 3, 4, 5, 2), (3, 4, 5, 2), None),
    ((2, 3, 4, 5, 2), (3, 4, 5, 2), (0,)),
    ((2, 3, 4, 5, 2), (3,), (0, 2, 3, 4))
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


@pytest.mark.parametrize('x_shape,reduced_shape,axis', _batch_norm_params)
@pytest.mark.parametrize('eps', [None, 3e-5, 1.2])
@pytest.mark.parametrize('decay', [None, 0.5])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_batch_norm(
        device, x_shape, reduced_shape, eps, decay, axis, float_dtype):
    def create_args(xp):
        return _create_batch_norm_ndarray_args(
            xp, device, x_shape, reduced_shape, reduced_shape, reduced_shape,
            reduced_shape, float_dtype)

    x_chx, gamma_chx, beta_chx, running_mean_chx, running_var_chx = (
        create_args(chainerx))
    x_np, gamma_np, beta_np, running_mean_np, running_var_np = (
        create_args(numpy))

    # Save copies of running values before updating to later check that they
    # are updated.
    initial_running_mean = running_mean_chx.copy()
    initial_running_var = running_var_chx.copy()

    optional_args = {}
    if eps is not None:
        optional_args['eps'] = eps
    if decay is not None:
        optional_args['decay'] = decay
    if axis is not None:
        optional_args['axis'] = axis

    y_chx = chainerx.batch_norm(
        x_chx, gamma_chx, beta_chx, running_mean=running_mean_chx,
        running_var=running_var_chx, **optional_args)
    y_np = chainer.functions.batch_normalization(
        x_np, gamma_np, beta_np, running_mean=running_mean_np,
        running_var=running_var_np, **optional_args).data

    # Check that the running values are updated.
    assert not numpy.allclose(chainerx.to_numpy(
        initial_running_mean), chainerx.to_numpy(running_mean_chx))
    assert not numpy.allclose(chainerx.to_numpy(
        initial_running_var), chainerx.to_numpy(running_var_chx))

    chainerx.testing.assert_allclose_ex(
        y_chx, y_np, rtol=1e-6, atol=1e-5,
        float16_rtol=1e-2, float16_atol=1e-2)
    chainerx.testing.assert_allclose_ex(
        running_mean_chx, running_mean_np,
        rtol=1e-6, atol=1e-6, float16_rtol=1e-2, float16_atol=1e-2)
    chainerx.testing.assert_allclose_ex(
        running_var_chx, running_var_np,
        rtol=1e-6, atol=1e-6, float16_rtol=1e-2, float16_atol=1e-2)


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


@pytest.mark.parametrize('x_shape,reduced_shape,axis', _batch_norm_params)
@pytest.mark.parametrize('eps', [None, 3e-5, 1.2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fixed_batch_norm(
        device, x_shape, reduced_shape, eps, axis, float_dtype):
    def create_args(xp):
        return _create_batch_norm_ndarray_args(
            xp, device, x_shape, reduced_shape, reduced_shape, reduced_shape,
            reduced_shape, float_dtype)

    x_chx, gamma_chx, beta_chx, mean_chx, var_chx = create_args(chainerx)
    x_np, gamma_np, beta_np, mean_np, var_np = create_args(numpy)

    optional_args = {}
    if eps is not None:
        optional_args['eps'] = eps
    if axis is not None:
        optional_args['axis'] = axis

    y_chx = chainerx.fixed_batch_norm(
        x_chx, gamma_chx, beta_chx, mean=mean_chx, var=var_chx,
        **optional_args)
    y_np = chainer.functions.fixed_batch_normalization(
        x_np, gamma_np, beta_np, mean=mean_np, var=var_np,
        **optional_args).data

    chainerx.testing.assert_allclose_ex(
        y_chx, y_np, rtol=1e-6, atol=1e-5,
        float16_rtol=1e-2, float16_atol=1e-2)


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
